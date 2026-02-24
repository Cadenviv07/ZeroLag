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
    // Use this method as the place to do any pre-playback
    // initialisation that you need..
    circularBuffer.setSize(getTotalNumInputChannels(), fftSize);
    olaBuffer.setSize(getTotalNumInputChannels(), fftSize*2);
    magnitude.setSize(getTotalNumInputChannels(), fftSize * 2);
    noiseFloor.setSize(getTotalNumInputChannels(), fftSize * 2);
    olaBuffer.clear();
    noiseFloor.clear();
    magnitude.clear();
    circularBuffer.clear();

    writePointer = 0;
    count = 0;
    calibrationCounter = 0;
    totalSamplesProcessed = 0;

    windowTable.assign(512, 0.0f);
    for (int n = 0; n < 512; ++n) {
        windowTable[n] = 0.5f * (1.0f - std::cos(2.0f * juce::MathConstants<float>::pi * n / 511.0f));
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
    auto totalNumOutputChannels = getTotalNumOutputChannels();

    for (auto i = totalNumInputChannels; i < totalNumOutputChannels; ++i)
        buffer.clear(i, 0, buffer.getNumSamples());

    // This is the place where you'd normally do the guts of your plugin's
    // audio processing...
    // Make sure to reset the state if your inner loop is processing
    // the samples and the outer loop is handling the channels.
    // Alternatively, you can process the samples with the channels
    // interleaved by keeping the same state.
    int startWritePos = writePointer;
    for (int channel = 0; channel < totalNumInputChannels; ++channel)
    {
        auto* channelData = buffer.getWritePointer(channel);
        int currentWritePos = startWritePos;
        // ..do something to the data...
        // i loops through the 64 samples while currentWritePos loops through the 512 frequencies in the ciruclar buffer
        for (int i = 0; i < buffer.getNumSamples(); i++) {
            circularBuffer.setSample(channel, currentWritePos, channelData[i]);

            // PULL FROM OLA: Give the host the processed sample from the conveyor belt
            channelData[i] = olaBuffer.getSample(channel, i);
            // Clear used sample
            olaBuffer.setSample(channel, i, 0.0f);

            currentWritePos++;
            currentWritePos &= (fftSize - 1);//Bitwise trick to reset it to zero once it reaches 512
        }
    }
    //Update writePointer for next function call
    writePointer = (startWritePos + buffer.getNumSamples()) & (fftSize - 1);

    count += buffer.getNumSamples();

    while (count >= shiftSize) {
        for (int channel = 0; channel < totalNumInputChannels; ++channel) {
            int fftPos = (writePointer - fftSize + fftSize) & (fftSize - 1);
            for (int sample = 0; sample < fftSize * 2; sample += 2) {
                fftBuffer[sample] = circularBuffer.getSample(channel, fftPos);
                fftBuffer[sample] *= windowTable[sample / 2];
                //Add an imaginary number for every real number sample
                fftBuffer[sample + 1] = 0.0f;
                fftPos += 1;
                fftPos &= (fftSize - 1);
            }
            //Finish with FFT data and copy it out to repeat for next channel
            auto* complexData = reinterpret_cast<juce::dsp::Complex<float>*>(fftBuffer);

            forwardFFT->perform(complexData, complexData, false);

            float* magData = magnitude.getWritePointer(channel);

            for (int i = 0; i < 1024; i += 8) {

                __m256 temp = _mm256_load_ps(&fftBuffer[i]);

                __m256 squared = _mm256_mul_ps(temp, temp);

                //Create a permutation of the magniutudes swapping imaginary and real numbers
                __m256 swapped = _mm256_permute_ps(squared, _MM_SHUFFLE(2, 3, 0, 1));

                __m256 combined = _mm256_add_ps(squared, swapped);

                __m256 final = _mm256_sqrt_ps(combined);
                //Create duplicate magniutdes so each imaginary and real value in fftbuffer are scaled the same
                _mm256_store_ps(magData + i, final);

            }

            float* nfData = noiseFloor.getWritePointer(channel);

            for (int i = 1; i < 1024; i++) {
                if (calibrationCounter < 20) {
                    nfData[i] = magData[i];
                }
                else {
                    //If the frequency is noise
                    if (magData[i] < nfData[i]) {
                        nfData[i] = (0.9f * nfData[i - 1] + (0.1f * magData[i]));
                    }
                    //If the frequency is signal
                    else {
                        nfData[i] = (0.999f * nfData[i - 1] + (0.001f * magData[i]));
                    }
                }
            }
            // The hann window causes the vector to be scaled three times more
            __m256 normalization = _mm256_set1_ps(1.0f / 1536.0f);
            //AVX2 process eight floats at a time
            for (int i = 0; i < 1024; i += 8) {
                __m256 ones = _mm256_set1_ps(1.0f);

                // Load 8 floats (4 complex numbers)
                __m256 temp = _mm256_load_ps(&fftBuffer[i]);

                __m256 noise = _mm256_load_ps(nfData + i);

                __m256 mag = _mm256_load_ps(magData + i);

                //Avoid dividing by zero
                __m256 eps = _mm256_set1_ps(1e-7f);

                __m256 SNR = _mm256_div_ps(mag, _mm256_add_ps(noise, eps));

                __m256 SNRDenom = _mm256_add_ps(SNR, ones);

                __m256 gain = _mm256_div_ps(SNR, SNRDenom);

                __m256 combinedGain = _mm256_mul_ps(gain, normalization);

                __m256 supression = _mm256_mul_ps(temp, combinedGain);

                // Store back into fftBuffer
                _mm256_store_ps(&fftBuffer[i], supression);
            }
            if (channel == 0) {
                DBG("FFT Bin 10: " << fftBuffer[10]);
            }
            if (channel == 0) {
                DBG("Noise Floor [10]: " << noiseFloor.getSample(0, 10));
            }
            inverseFFT->perform(complexData, complexData, true);

            //Add the fading out audio of the fft buffer to the fading in audio of the enxt window
            for (int i = 0; i < 512; i++) {
                float processedSample = fftBuffer[i * 2] * windowTable[i];
                olaBuffer.setSample(channel, i, olaBuffer.getSample(channel, i) + processedSample);
            }
            if (channel == 0) {
                DBG("OLA Out: " << olaBuffer.getSample(0, 50));
            }
        }

        // Shift OLA buffer by shiftSize (64)
        for (int ch = 0; ch < totalNumInputChannels; ++ch) {
            for (int i = 0; i < (fftSize * 2) - shiftSize; ++i) {
                olaBuffer.setSample(ch, i, olaBuffer.getSample(ch, i + shiftSize));
            }
            olaBuffer.clear(ch, (fftSize * 2) - shiftSize, shiftSize);
        }

        count -= shiftSize;
        totalSamplesProcessed += shiftSize;
        calibrationCounter++;
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
