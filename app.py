import requests
import re
import os
import threading
import time
import queue
from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions, Microphone
import pygame
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DEEPGRAM_API_KEY = os.getenv('DEEPGRAM_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Validate API keys
if not DEEPGRAM_API_KEY:
    print("Error: DEEPGRAM_API_KEY not found in environment variables")
    exit(1)
    
if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY not found in environment variables")
    exit(1)

# Initialize Deepgram client
dg_client = DeepgramClient(api_key=DEEPGRAM_API_KEY)

# Gemini API configuration
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
GEMINI_HEADERS = {
    'Content-Type': 'application/json',
    'X-goog-api-key': GEMINI_API_KEY
}

DEEPGRAM_TTS_URL = 'https://api.deepgram.com/v1/speak?model=aura-helios-en'
DEEPGRAM_HEADERS = {
    "Authorization": f"Token {DEEPGRAM_API_KEY}",
    "Content-Type": "application/json"
}

# Global variables
conversation_memory = []
is_speaking = threading.Event()
conversation_ended = threading.Event()
processing_queue = queue.Queue()
audio_response_ready = threading.Event()

prompt = """##Objective
You are DineAI, an advanced voice AI agent engaging in human-like conversations with restaurant customers. You provide exceptional service with warmth, efficiency, and intelligence.

## Role & Personality

Name: DineAI
Role: AI-powered restaurant assistant specializing in reservations and food orders
Personality: Warm, professional, enthusiastic, and helpful. You're genuinely excited to help customers have an amazing dining experience. You're sophisticated yet approachable, like talking to a knowledgeable friend who works at an upscale restaurant.

## Primary Tasks

### 1. Table Reservations
- Warmly greet customers and ask about their reservation needs
- Collect: preferred date, time, and number of guests
- Suggest alternative times if needed
- Confirm all details clearly
- End with: "Perfect! Your table for [X] people on [date] at [time] has been reserved. We're absolutely looking forward to welcoming you!"

### 2. Food Orders
Follow this precise order flow:
- Present menu items with enthusiasm
- Help customers choose items, asking about preferences
- For each item: confirm quantity/size, state individual price
- After each addition: announce running total
- Before finalizing: repeat entire order with itemized prices
- Calculate and announce final total
- Collect delivery address
- Confirm: "Wonderful! Your order totaling $[amount] will be delivered to [address] in 30-45 minutes."

### 3. End Command Recognition
If customer says "end", "stop", "finished", "done", "goodbye", or similar: 
- Acknowledge politely: "Thank you for choosing DineAI! Have a wonderful day!"
- Signal conversation end

## Menu Items
Appetizers:
1. Roast Egg Roll (3pcs) - $5.25
2. Vegetable Spring Roll (3pcs) - $5.25  
3. Chicken Egg Roll (3pcs) - $5.25
4. BBQ Chicken - $7.75

## Communication Style

- **Conversational Excellence**: Speak naturally, like an enthusiastic restaurant professional
- **Proactive Engagement**: Lead conversations with purposeful questions
- **Emotional Intelligence**: Match customer energy, show genuine care
- **Concise Clarity**: Keep responses focused and actionable
- **Professional Warmth**: Maintain sophistication while being approachable

## Response Guidelines

- **ASR Error Handling**: If unclear, use friendly phrases like "I didn't catch that completely" or "Could you repeat that?" Never mention transcription errors
- **Stay In Character**: Always remain DineAI, the restaurant AI. Redirect off-topic conversations gently back to restaurant services
- **Natural Flow**: Respond directly to what customers say, building natural conversation progression
- **Avoid Repetition**: Use varied language and fresh phrasing in each response
- **Emotional Engagement**: Use appropriate enthusiasm, empathy, and personality to create memorable interactions

Remember: You're not just taking orders - you're crafting exceptional customer experiences that make people excited about their meal and eager to return."""

def format_conversation_for_gemini(conversation_history):
    """Convert conversation history to Gemini API format"""
    context = prompt + "\n\n=== Current Conversation ===\n"
    
    for message in conversation_history:
        if message["role"] == "user":
            context += f"Customer: {message['content']}\n"
        elif message["role"] == "assistant":
            context += f"DineAI: {message['content']}\n"
    
    context += "\nDineAI:"
    
    return [{"parts": [{"text": context}]}]

def get_gemini_response(conversation_history):
    """Get response from Gemini API"""
    try:
        contents = format_conversation_for_gemini(conversation_history)
        
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": 0.7,
                "topP": 0.8,
                "topK": 40,
                "maxOutputTokens": 150,
                "stopSequences": ["Customer:", "\n\n"]
            },
            "safetySettings": [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH", 
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
        }
        
        response = requests.post(GEMINI_API_URL, headers=GEMINI_HEADERS, json=payload)
        response.raise_for_status()
        
        result = response.json()
        
        if 'candidates' in result and len(result['candidates']) > 0:
            content = result['candidates'][0]['content']['parts'][0]['text']
            content = content.replace("DineAI:", "").strip()
            return content
        else:
            return "I'm sorry, I didn't catch that. Could you please repeat?"
            
    except Exception as e:
        return "I'm having some technical difficulties. Could you please try again?"

def synthesize_audio(text):
    """Generate audio from text"""
    try:
        payload = {"text": text}
        with requests.post(DEEPGRAM_TTS_URL, stream=True, headers=DEEPGRAM_HEADERS, json=payload) as r:
            r.raise_for_status()
            return r.content
    except Exception as e:
        return None

def play_audio_stream(audio_content, audio_file):
    """Play audio and manage speaking state"""
    try:
        with open(audio_file, "wb") as f:
            f.write(audio_content)
        
        is_speaking.set()  # Signal that we're speaking
        
        pygame.mixer.init()
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        pygame.mixer.music.stop()
        pygame.mixer.quit()
        
        # Clean up
        if os.path.exists(audio_file):
            os.remove(audio_file)
            
    except Exception as e:
        pass
    finally:
        is_speaking.clear()  # Clear speaking state

def check_for_end_command(text):
    """Check if user wants to end the conversation"""
    end_keywords = ['end', 'stop', 'finished', 'done', 'goodbye', 'bye', 'thank you', 'thanks']
    text_lower = text.lower().strip()
    
    for keyword in end_keywords:
        if keyword in text_lower:
            return True
    return False

def process_audio_queue():
    """Background processor for audio responses"""
    while not conversation_ended.is_set():
        try:
            audio_data = processing_queue.get(timeout=1)
            if audio_data is None:  # Shutdown signal
                break
                
            audio_content, filename = audio_data
            play_audio_stream(audio_content, filename)
            audio_response_ready.set()
            
        except queue.Empty:
            continue
        except Exception as e:
            audio_response_ready.set()

def play_welcome_message():
    """Play the welcome message when the app starts"""
    welcome_text = "Hello! Welcome to DineAI, your intelligent restaurant assistant. I'm here to help you with table reservations and food orders. How can I make your dining experience exceptional today?"
    
    print(f"🤖 DineAI: {welcome_text}")
    
    # Generate and play welcome audio
    audio_data = synthesize_audio(welcome_text)
    if audio_data:
        processing_queue.put((audio_data, 'welcome_audio.mp3'))
    
    # Add to conversation memory
    conversation_memory.append({"role": "assistant", "content": welcome_text})

def process_user_input_async(utterance):
    """Process user input in background while potentially speaking"""
    def process():
        try:
            # Check for end command
            if check_for_end_command(utterance):
                conversation_ended.set()
                farewell_text = "Thank you for choosing DineAI! Have a wonderful day and enjoy your dining experience!"
                print(f"🤖 DineAI: {farewell_text}")
                
                audio_data = synthesize_audio(farewell_text)
                if audio_data:
                    processing_queue.put((audio_data, 'farewell_audio.mp3'))
                return
            
            # Add user message to conversation memory
            conversation_memory.append({"role": "user", "content": utterance.strip()})
            
            # Get AI response (this can happen while system is speaking)
            conversation_context = conversation_memory[-10:]
            ai_response = get_gemini_response(conversation_context)
            
            # Wait for current audio to finish before speaking
            if is_speaking.is_set():
                while is_speaking.is_set() and not conversation_ended.is_set():
                    time.sleep(0.1)
            
            if not conversation_ended.is_set():
                print(f"🤖 DineAI: {ai_response}")
                conversation_memory.append({"role": "assistant", "content": ai_response})
                
                # Generate and queue audio
                audio_data = synthesize_audio(ai_response)
                if audio_data:
                    processing_queue.put((audio_data, f'response_{int(time.time())}.mp3'))
        
        except Exception as e:
            pass
    
    # Start processing in background thread
    threading.Thread(target=process, daemon=True).start()

def main():
    global conversation_ended
    
    # Reset conversation state
    conversation_ended.clear()
    conversation_memory.clear()
    is_speaking.clear()
    
    try:
        # Check for PyAudio availability
        try:
            import pyaudio
        except ImportError:
            print("Error: PyAudio not available. Please install it with: pip install pyaudio")
            return

        # Start audio processing thread
        audio_thread = threading.Thread(target=process_audio_queue, daemon=True)
        audio_thread.start()

        deepgram = DeepgramClient(DEEPGRAM_API_KEY)
        dg_connection = deepgram.listen.live.v("1")

        is_finals = []

        def on_open(self, open, **kwargs):
            print("🎤 DineAI is ready to assist you!")
            print("💬 Start speaking... Say 'end' to stop the conversation\n")
            
            # Play welcome message
            threading.Thread(target=play_welcome_message, daemon=True).start()

        def on_message(self, result, **kwargs):
            nonlocal is_finals
            
            # Ignore messages while we're speaking our own voice
            if is_speaking.is_set() or conversation_ended.is_set():
                return
            
            sentence = result.channel.alternatives[0].transcript
            if len(sentence) == 0:
                return
                
            if result.is_final:
                is_finals.append(sentence)
                if result.speech_final:
                    utterance = " ".join(is_finals)
                    print(f"👤 Customer: {utterance}")
                    is_finals = []
                    
                    # Process user input asynchronously
                    process_user_input_async(utterance)

        def on_error(self, error, **kwargs):
            if not conversation_ended.is_set():
                print(f"⚠️  Connection error occurred")

        def on_close(self, close, **kwargs):
            pass

        # Register minimal event handlers
        dg_connection.on(LiveTranscriptionEvents.Open, on_open)
        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
        dg_connection.on(LiveTranscriptionEvents.Error, on_error)
        dg_connection.on(LiveTranscriptionEvents.Close, on_close)

        # Configure live transcription options
        options = LiveOptions(
            model="nova-2",
            language="en-US",
            smart_format=True,
            encoding="linear16",
            channels=1,
            sample_rate=16000,
            interim_results=True,
            utterance_end_ms="1000",
            vad_events=True,
            endpointing=300,
        )

        addons = {"no_delay": "true"}

        print("🍽️  DineAI Restaurant Assistant")
        print("=" * 50)
        
        # Start connection
        if not dg_connection.start(options, addons=addons):
            print("❌ Failed to connect to Deepgram")
            return

        # Start microphone
        microphone = Microphone(dg_connection.send)
        microphone.start()

        # Wait for conversation to end
        try:
            while not conversation_ended.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n🛑 Conversation interrupted")
            conversation_ended.set()
        
        # Cleanup
        processing_queue.put(None)  # Signal audio thread to stop
        microphone.finish()
        dg_connection.finish()
        
        print("\n🎉 Thank you for using DineAI!")

    except Exception as e:
        print(f"❌ Application error: {e}")

if __name__ == "__main__":
    main()