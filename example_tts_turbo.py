import argparse
import platform
import shutil
import subprocess
import time
from pathlib import Path

import torchaudio as ta

from chatterbox.tts_turbo import ChatterboxTurboTTS

DEFAULT_TEXT = (
    "Oh, that's hilarious! [chuckle] Um anyway, we do have a new model in store. "
    "It's the SkyNet T-800 series and it's got basically everything. "
    "Including AI integration with ChatGPT and all that jazz. "
    "Would you like me to get some prices for you?"
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate TTS audio using ChatterboxTurbo"
    )
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Text to synthesize")
    parser.add_argument("--text-file", dest="text_file", type=Path, help="Read text from file")
    parser.add_argument(
        "--audio-prompt", dest="audio_prompt", help="Reference audio for voice cloning"
    )
    parser.add_argument(
        "--output", "-o", default="test-turbo.wav", help="Output wav file path"
    )
    default_device = "mps" if platform.system() == "Darwin" else "cuda"
    parser.add_argument(
        "--device", default=default_device, help="Device to run on (mps, cuda, cpu)"
    )
    parser.add_argument(
        "--play", action="store_true", help="Auto-play audio after generation"
    )
    parser.add_argument(
        "--metrics", action="store_true", help="Display generation performance stats"
    )
    args = parser.parse_args()

    text = args.text_file.read_text() if args.text_file else args.text

    model = ChatterboxTurboTTS.from_pretrained(device=args.device)

    generate_kwargs = {}
    if args.audio_prompt:
        generate_kwargs["audio_prompt_path"] = args.audio_prompt

    start_time = time.perf_counter()
    wav = model.generate(text, **generate_kwargs)
    gen_time = time.perf_counter() - start_time

    ta.save(args.output, wav, model.sr)
    print(f"Saved to {args.output}")

    if args.metrics:
        audio_duration = wav.shape[1] / model.sr
        rtf = gen_time / audio_duration
        print(f"Generation time: {gen_time:.2f}s | Audio duration: {audio_duration:.2f}s | RTF: {rtf:.2f}x")

    if args.play:
        # Try common audio players in order of preference
        players = ["afplay", "ffplay", "aplay", "paplay"]
        for player in players:
            if shutil.which(player):
                cmd = [player]
                if player == "ffplay":
                    cmd.extend(["-nodisp", "-autoexit"])
                cmd.append(args.output)
                subprocess.run(cmd)
                break
        else:
            print("No audio player found (tried: ffplay, afplay, aplay, paplay)")


if __name__ == "__main__":
    main()
