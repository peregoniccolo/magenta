import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow

import sounddevice as sd
import soundfile as sf
import librosa
from magenta.models.gansynth.lib import flags as lib_flags
from magenta.models.gansynth.lib import generate_util as gu
from magenta.models.gansynth.lib import model as lib_model
from magenta.models.gansynth.lib import util
import matplotlib.pyplot as plt
import note_seq
import numpy as np
import tkinter as tk
from tkinter import filedialog
import time


# GLOBALS
CKPT_DIR = "gs://magentadata/models/gansynth/acoustic_only"
output_dir = "gansynth/samples"
output_name = "generated_audio"
BATCH_SIZE = 16
SR = 16000


# ----------- Helper functions -----------
def check_out_dir():
	"""Make an output directory if it doesn't exist"""
	OUTPUT_DIR = util.expand_path(output_dir)
	if not os.path.exists(OUTPUT_DIR):
		os.makedirs(OUTPUT_DIR)


def load_model():
	"""Load the model"""
	tf.reset_default_graph()
	flags = lib_flags.Flags(
		{
			"batch_size_schedule": [BATCH_SIZE],
			"tfds_data_dir": "gs://tfds-data/datasets",
		}
	)
	model = lib_model.Model.load_from_path(CKPT_DIR, flags)
	return model


def load_midi(midi_path, min_pitch=36, max_pitch=84):
	"""Load midi as a notesequence."""
	midi_path = util.expand_path(midi_path)
	ns = note_seq.midi_file_to_sequence_proto(midi_path)
	pitches = np.array([n.pitch for n in ns.notes])
	velocities = np.array([n.velocity for n in ns.notes])
	start_times = np.array([n.start_time for n in ns.notes])
	end_times = np.array([n.end_time for n in ns.notes])
	valid = np.logical_and(pitches >= min_pitch, pitches <= max_pitch)
	notes = {
		"pitches": pitches[valid],
		"velocities": velocities[valid],
		"start_times": start_times[valid],
		"end_times": end_times[valid],
	}
	return ns, notes


def get_envelope(t_note_length, t_attack=0.010, t_release=0.3, sr=16000):
	"""Create an attack sustain release amplitude envelope."""
	t_note_length = min(t_note_length, 3.0)
	i_attack = int(sr * t_attack)
	i_sustain = int(sr * t_note_length)
	i_release = int(sr * t_release)
	i_tot = i_sustain + i_release  # attack envelope doesn't add to sound length
	envelope = np.ones(i_tot)
	# Linear attack
	envelope[:i_attack] = np.linspace(0.0, 1.0, i_attack)
	# Linear release
	envelope[i_sustain:i_tot] = np.linspace(1.0, 0.0, i_release)
	return envelope


def combine_notes(audio_notes, start_times, end_times, velocities, sr=16000):
	"""Combine audio from multiple notes into a single audio clip.

	Args:
	  audio_notes: Array of audio [n_notes, audio_samples].
	  start_times: Array of note starts in seconds [n_notes].
	  end_times: Array of note ends in seconds [n_notes].
	  sr: Integer, sample rate.

	Returns:
	  audio_clip: Array of combined audio clip [audio_samples]
	"""
	n_notes = len(audio_notes)
	clip_length = end_times.max() + 3.0
	audio_clip = np.zeros(int(clip_length) * sr)

	for t_start, t_end, vel, i in zip(
		start_times, end_times, velocities, range(n_notes)
	):
		# Generate an amplitude envelope
		t_note_length = t_end - t_start
		envelope = get_envelope(t_note_length)
		length = len(envelope)
		audio_note = audio_notes[i, :length] * envelope
		# Normalize
		audio_note /= audio_note.max()
		audio_note *= vel / 127.0
		# Add to clip buffer
		clip_start = int(t_start * sr)
		clip_end = clip_start + length
		audio_clip[clip_start:clip_end] += audio_note

	# Normalize
	audio_clip /= audio_clip.max()
	audio_clip /= 2.0
	return audio_clip


def specplot(audio_clip):
	"""Plots spectrogram of audio_clip"""
	p_min = np.min(36)
	p_max = np.max(84)
	f_min = librosa.midi_to_hz(p_min)
	f_max = 2 * librosa.midi_to_hz(p_max)
	octaves = int(np.ceil(np.log2(f_max) - np.log2(f_min)))
	bins_per_octave = 36
	n_bins = int(bins_per_octave * octaves)
	C = librosa.cqt(
		audio_clip,
		sr=SR,
		hop_length=2048,
		fmin=f_min,
		n_bins=n_bins,
		bins_per_octave=bins_per_octave,
	)
	power = 10 * np.log10(np.abs(C) ** 2 + 1e-6)
	plt.matshow(power[::-1, 2:-2], aspect="auto", cmap=plt.cm.magma)
	plt.yticks([])
	plt.xticks([])


def select_midi_seq():
	root = tk.Tk()
	root.withdraw()

	file_path = filedialog.askopenfilename()
	return file_path


def get_num_instruments():
	number = input("Select the number of instruments to generate: ")
	while not number.isdigit():
		number = input(
			"Input wasn't a digit, please select the number of instruments to generate: "
		)
	if int(number) == 0:
		print("Number must be > 0")
		number = get_num_instruments()
	return int(number)


def play_audio_array(to_play):
	sd.play(to_play, SR)
	time.sleep(len(to_play) / SR)
	sd.stop()


def setup():
	global tf
	tf = tensorflow.compat.v1
	tf.disable_v2_behavior()
	tf.logging.set_verbosity(tf.logging.ERROR)

	print("starting setup")
	check_out_dir()
	model = load_model()
	print("done loading\n")
	return model


def generate_instruments(model, midi_path, number_of_random_instr_seq):
	# load midi file
	ns = None
	try:
		ns, notes = load_midi(midi_path)
	except Exception as e:
		print("Failed loading MIDI")
		exit(1)
	# while ns == None:
	# 	midi_path = select_midi_seq()
	# 	try:
	# 		ns, notes = load_midi(midi_path)
	# 	except Exception as e:
	# 		print('Failed loading MIDI, try again')

	print(f"Loaded {midi_path}")
	# note_seq.plot_sequence(ns)

	# sample latent space for random instr_seq
	pitch_preview = 60
	n_preview = number_of_random_instr_seq

	pitches_preview = [pitch_preview] * n_preview
	print("generating...")
	z_preview = model.generate_z(n_preview)

	audio_notes = model.generate_samples_from_z(z_preview, pitches_preview)
	audio_notes_list = []
	for i, audio_note in enumerate(audio_notes):
		print(f"Instrument: {i}")
		audio_notes_list.append(audio_note)

	return audio_notes_list, z_preview, notes


def generate_audio(
	model,
	z_preview,
	notes,
	instr_seq=[0, 1, 2],
	time_seq=[0, 0.5, 1.0],
	name=output_name,
):
	# Force endpoints
	time_seq[0] = -0.001
	time_seq[-1] = 1.0

	z_instr_seq = np.array([z_preview[i] for i in instr_seq])
	t_instr_seq = np.array([notes["end_times"][-1] * t for t in time_seq])

	# Get latent vectors for each note
	z_notes = gu.get_z_notes(notes["start_times"], z_instr_seq, t_instr_seq)

	# Generate audio for each note
	print(f"Generating {len(z_notes)} samples...")
	audio_notes = model.generate_samples_from_z(z_notes, notes["pitches"])

	# Make a single audio clip
	audio_clip = combine_notes(
		audio_notes, notes["start_times"], notes["end_times"], notes["velocities"]
	)

	# Play the audio
	# print('\nPlaying generated audio...')
	# play_audio_array(audio_clip)

	fname = os.path.join(output_dir, f"{name}.mp3")
	sf.write(data=audio_clip, file=fname, samplerate=SR)
	print(f"saved at {fname}")

	time_seq[0] = 0
	time_seq[-1] = 1


def set_output_dir(dir):
	global output_dir
	output_dir = dir


if __name__ == "__main__":
	model = setup()
	num = get_num_instruments()
	midi_path = select_midi_seq()
	audio_notes_list, z_preview, notes = generate_instruments(model, midi_path, num)
	instr_seq = [0, 1, 2, 0]
	time_seq = [0, 0.3, 0.6, 1.0]
	generate_audio(model, z_preview, notes, instr_seq, time_seq, "test_name")
