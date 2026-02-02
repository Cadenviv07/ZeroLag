/*
  ==============================================================================

    This file contains the basic framework code for a JUCE plugin processor.

  ==============================================================================
*/

#include "PluginProcessor.h"
#include "PluginEditor.h"
#include <memory>
#include <juce_dsp/juce_dsp.h>

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

void ZeroLagAudioProcessor::processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages)
{
    juce::ScopedNoDenormals noDenormals;
    auto totalNumInputChannels  = getTotalNumInputChannels();
    auto totalNumOutputChannels = getTotalNumOutputChannels();

    // In case we have more outputs than inputs, this code clears any output
    // channels that didn't contain input data, (because these aren't
    // guaranteed to be empty - they may contain garbage).
    // This is here to avoid people getting screaming feedback
    // when they first compile a plugin, but obviously you don't need to keep
    // this code if your algorithm always overwrites all the output channels.
    for (auto i = totalNumInputChannels; i < totalNumOutputChannels; ++i)
        buffer.clear (i, 0, buffer.getNumSamples());

    // This is the place where you'd normally do the guts of your plugin's
    // audio processing...
    // Make sure to reset the state if your inner loop is processing
    // the samples and the outer loop is handling the channels.
    // Alternatively, you can process the samples with the channels
    // interleaved by keeping the same state.
    int startWritePos = writePointer;
    for (int channel = 0; channel < totalNumInputChannels; ++channel)
    {
        auto* channelData = buffer.getWritePointer (channel);
        int currentWritePos = startWritePos;
        // ..do something to the data...
        // i loops through the 64 samples while currentWritePos loops through the 512 frequencies in the ciruclar buffer
        for (int i = 0; i < buffer.getNumSamples(); i++) {
            circularBuffer.setSample(channel, currentWritePos, channelData[i]);
            currentWritePos++;
            currentWritePos &= (fftSize - 1);//Bitwise trick to reset it to zero once it reaches 512
        }
    }
    //Update writePointer for next function call
    writePointer = (startWritePos + buffer.getNumSamples()) & (fftSize - 1);

    count += buffer.getNumSamples();

   
    if (count >= shiftSize && count > fftSize) {
        for (int channel = 0; channel < totalNumInputChannels; ++channel) {
            int fftPos = writePointer;
            for (int sample = 0; sample < fftSize * 2; sample += 2) {
                fftBuffer[sample] = circularBuffer.getSample(channel, fftPos);
                //Add an imaginary number for every real number sample
                fftBuffer[sample + 1] = 0.0f;
                fftPos += 1;
                fftPos &= (fftSize - 1);
            }
            //Finish with FFT data and copy it out to repeat for next channel
            auto* complexData = reinterpret_cast<juce::dsp::Complex<float>*>(fftBuffer);

            forwardFFT->perform (complexData, complexData, false);
            inverseFFT->perform(complexData, complexData, true);
        }
        count == 0;
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
