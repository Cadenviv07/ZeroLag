/*
  ==============================================================================

    This file contains the basic framework code for a JUCE plugin processor.

  ==============================================================================
*/

#include "PluginProcessor.h"
#include "PluginEditor.h"
#include <memory>
#include <juce_dsp/juce_dsp.h>
#include <immintrin.h>

//==============================================================================

ZeroLagAudioProcessor::ZeroLagAudioProcessor()
#ifndef JucePlugin_PreferredChannelConfigurations
     : AudioProcessor (BusesProperties()
                     #if ! JucePlugin_IsMidiEffect
                      #if ! JucePlugin_IsSynth
                       .withInput  ("Input",  juce::AudioChannelSet::stereo(), true)
                      #endif
                       .withOutput ("Output", juce::AudioChannelSet::stereo(), true)
                     #endif
                       ),
    forwardFFT(std::make_unique<juce::dsp::FFT>(9)), // Note the comma above
    inverseFFT(std::make_unique<juce::dsp::FFT>(9))
#endif
{
}

ZeroLagAudioProcessor::~ZeroLagAudioProcessor()
{
}

//==============================================================================
const juce::String ZeroLagAudioProcessor::getName() const
{
    return JucePlugin_Name;
}

bool ZeroLagAudioProcessor::acceptsMidi() const
{
   #if JucePlugin_WantsMidiInput
    return true;
   #else
    return false;
   #endif
}

bool ZeroLagAudioProcessor::producesMidi() const
{
   #if JucePlugin_ProducesMidiOutput
    return true;
   #else
    return false;
   #endif
}

bool ZeroLagAudioProcessor::isMidiEffect() const
{
   #if JucePlugin_IsMidiEffect
    return true;
   #else
    return false;
   #endif
}

double ZeroLagAudioProcessor::getTailLengthSeconds() const
{
    return 0.0;
}

int ZeroLagAudioProcessor::getNumPrograms()
{
    return 1;   // NB: some hosts don't cope very well if you tell them there are 0 programs,
                // so this should be at least 1, even if you're not really implementing programs.
}

int ZeroLagAudioProcessor::getCurrentProgram()
{
    return 0;
}

void ZeroLagAudioProcessor::setCurrentProgram (int index)
{
}

const juce::String ZeroLagAudioProcessor::getProgramName (int index)
{
    return {};
}

void ZeroLagAudioProcessor::changeProgramName (int index, const juce::String& newName)
{
}

//==============================================================================
void ZeroLagAudioProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    circularBuffer.setSize(getTotalNumInputChannels(), 8192);
    olaBuffer.setSize(getTotalNumInputChannels(), 512);

    
    magnitude.setSize(getTotalNumInputChannels(), 1024);
    noiseFloor.setSize(getTotalNumInputChannels(), 1024);

    olaBuffer.clear();
    noiseFloor.clear();
    magnitude.clear();
    circularBuffer.clear();

    writePointer = 0;
    count = 0;
    calibrationCounter = 0;
    totalSamplesProcessed = 0;

    windowTable.assign(512, 0.0f);
    const float normFactor = 1.0f / 2048.0f;
    for (int n = 0; n < 512; ++n) {
        windowTable[n] = normFactor * 0.5f * (1.0f - std::cos(2.0f * juce::MathConstants<float>::pi * n / 512.0f));
    }
}

void ZeroLagAudioProcessor::releaseResources()
{
    // When playback stops, you can use this as an opportunity to free up any
    // spare memory, etc.
}

#ifndef JucePlugin_PreferredChannelConfigurations
bool ZeroLagAudioProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
  #if JucePlugin_IsMidiEffect
    juce::ignoreUnused (layouts);
    return true;
  #else
    // This is the place where you check if the layout is supported.
    // In this template code we only support mono or stereo.
    // Some plugin hosts, such as certain GarageBand versions, will only
    // load plugins that support stereo bus layouts.
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::mono()
     && layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
        return false;

    // This checks if the input layout matches the output layout
   #if ! JucePlugin_IsSynth
    if (layouts.getMainOutputChannelSet() != layouts.getMainInputChannelSet())
        return false;
   #endif

    return true;
  #endif
}
#endif


void ZeroLagAudioProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages)
{
    juce::ScopedNoDenormals noDenormals;
    auto totalNumInputChannels = getTotalNumInputChannels();

    // Clear extra output channels
    for (auto i = totalNumInputChannels; i < getTotalNumOutputChannels(); ++i)
        buffer.clear(i, 0, buffer.getNumSamples());

    // THE SAMPLE-BY-SAMPLE STATE MACHINE
    // This perfectly decouples your 64-hop DSP from the DAW's block size.
    for (int i = 0; i < buffer.getNumSamples(); ++i) {

        // 1. Output a valid processed sample, then save the dry sample
        for (int channel = 0; channel < totalNumInputChannels; ++channel) {
            float drySample = buffer.getSample(channel, i);

            // Output exactly 1 sample of finished audio from the OLA buffer
            buffer.setSample(channel, i, olaBuffer.getSample(channel, count));

            // Push the dry audio to the circular history buffer
            circularBuffer.setSample(channel, writePointer, drySample);
        }

        writePointer = (writePointer + 1) & 8191; // Wrap safely at 8192
        count++;
        totalSamplesProcessed++;

        // 2. Trigger the heavy math exactly every 64 samples
        if (count >= 64) {
            for (int channel = 0; channel < totalNumInputChannels; ++channel) {

                // Backtrack exactly 512 samples in the circular buffer
                int fftPos = (writePointer - 512 + 8192) & 8191;

                // Pack the interleaved complex array
                for (int sample = 0; sample < 1024; sample += 2) {
                    fftBuffer[sample] = circularBuffer.getSample(channel, fftPos);
                    fftBuffer[sample] *= windowTable[sample / 2];
                    fftBuffer[sample + 1] = 0.0f;
                    fftPos = (fftPos + 1) & 8191;
                }

                auto* complexData = reinterpret_cast<juce::dsp::Complex<float>*>(fftBuffer);
                forwardFFT->perform(complexData, complexData, false);



                //float* magData = magnitude.getWritePointer(channel);

                ////for (int i = 0; i < 1024; i += 8) {

                ////    __m256 temp = _mm256_load_ps(&fftBuffer[i]);

                ////    __m256 squared = _mm256_mul_ps(temp, temp);

                ////    //Create a permutation of the magniutudes swapping imaginary and real numbers
                ////    __m256 swapped = _mm256_permute_ps(squared, _MM_SHUFFLE(2, 3, 0, 1));

                ////    __m256 combined = _mm256_add_ps(squared, swapped);

                ////    __m256 final = _mm256_sqrt_ps(combined);
                ////    //Create duplicate magniutdes so each imaginary and real value in fftbuffer are scaled the same
                ////    _mm256_store_ps(magData + i, final);

                ////}

                //float* nfData = noiseFloor.getWritePointer(channel);

                //for (int i = 1; i < 1024; i++) {
                //    if (calibrationCounter < 20) {
                //        nfData[i] = magData[i];
                //    }
                //    else {
                //        //If the frequency is noise
                //        if (magData[i] < nfData[i]) {
                //            nfData[i] = (0.9f * nfData[i] + (0.1f * magData[i]));
                //        }
                //        //If the frequency is signal
                //        else {
                //            nfData[i] = (0.999f * nfData[i] + (0.001f * magData[i]));
                //        }
                //    }
                //}
                // The hann window causes the vector to be scaled three times more
                //__m256 normalization = _mm256_set1_ps(1.0f / 1536.0f);
                //AVX2 process eight floats at a time
                //for (int i = 0; i < 1024; i += 8) {
                //    __m256 ones = _mm256_set1_ps(1.0f);

                //    // Load 8 floats (4 complex numbers)
                //    __m256 temp = _mm256_load_ps(&fftBuffer[i]);

                //    __m256 noise = _mm256_load_ps(nfData + i);

                //    __m256 mag = _mm256_load_ps(magData + i);

                //    //Avoid dividing by zero
                //    __m256 eps = _mm256_set1_ps(1e-7f);

                //    __m256 SNR = _mm256_div_ps(mag, _mm256_add_ps(noise, eps));

                //    __m256 SNRDenom = _mm256_add_ps(SNR, ones);

                //    __m256 gain = _mm256_div_ps(SNR, SNRDenom);

                //    //__m256 combinedGain = _mm256_mul_ps(gain, normalization);

                //    __m256 supression = _mm256_mul_ps(temp, gain);

                //    //Store back into fftBuffer
                //    _mm256_store_ps(&fftBuffer[i], supression);
                //}

                inverseFFT->perform(complexData, complexData, true);

                // Add the new IFFT frame directly into the OLA buffer
                for (int j = 0; j < 512; j++) {
                    float processedRealSample = fftBuffer[j * 2];
                    float currentOlaSample = olaBuffer.getSample(channel, j);
                    olaBuffer.setSample(channel, j, currentOlaSample + processedRealSample);
                }
            }

            // 3. Shift the OLA buffer by 64 and clear the tail
            for (int ch = 0; ch < totalNumInputChannels; ++ch) {
                for (int j = 0; j < 512 - 64; ++j) {
                    olaBuffer.setSample(ch, j, olaBuffer.getSample(ch, j + 64));
                }
                olaBuffer.clear(ch, 512 - 64, 64);
            }

            count = 0; // Reset the 64-sample trigger
            calibrationCounter++;
        }
    }
}
//==============================================================================
bool ZeroLagAudioProcessor::hasEditor() const
{
    return true; // (change this to false if you choose to not supply an editor)
}

juce::AudioProcessorEditor* ZeroLagAudioProcessor::createEditor()
{
    return new ZeroLagAudioProcessorEditor (*this);
}

//==============================================================================
void ZeroLagAudioProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    // You should use this method to store your parameters in the memory block.
    // You could do that either as raw data, or use the XML or ValueTree classes
    // as intermediaries to make it easy to save and load complex data.
}

void ZeroLagAudioProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    // You should use this method to restore your parameters from this memory block,
    // whose contents will have been created by the getStateInformation() call.
}

//==============================================================================
// This creates new instances of the plugin..
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new ZeroLagAudioProcessor();
}
