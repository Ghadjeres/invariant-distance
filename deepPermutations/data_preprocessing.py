import os
import pickle
from typing import List, Optional

from DeepBach.metadata import *
from music21 import converter, interval, corpus, stream, note, duration
from music21.analysis.floatingKey import FloatingKeyException
from tqdm import tqdm

from .data_utils import create_index_dicts, _min_max_midi_pitch, \
    chorale_to_inputs, compute_min_max_pitches, \
    filter_file_list, SLUR_SYMBOL, standard_note

SOP_INDEX = 0


def initialize_transposition_dataset(dataset_dir=None,
                                     metadatas: List[Metadata] = []):
    """
    Create 'datasets/transpose/bach_sop.pickle' or
    'datasets/transpose/custom_dataset.pickle'
    :param dataset_dir: use Bach chorales if None
    :param metadatas:
    :type metadatas: Metadata
    :return:
    :rtype:
    """
    from glob import glob
    PACKAGE_DIR = os.path.dirname(__file__)
    NUM_VOICES = 1
    voice_ids = [SOP_INDEX]
    print('Creating dataset')
    if dataset_dir:
        chorale_list = filter_file_list(
            glob(dataset_dir + '/*.mid') + glob(dataset_dir + '/*.xml'),
            num_voices=NUM_VOICES)
        pickled_dataset = os.path.join(PACKAGE_DIR,
                                       'datasets/transpose/' + \
                                       dataset_dir.split(
                                           '/')[
                                           -1] + '.pickle')
    else:
        chorale_list = filter_file_list(
            corpus.getBachChorales(fileExtensions='xml'))
        pickled_dataset = os.path.join(PACKAGE_DIR,
                                       'datasets/transpose/bach_sop.pickle'
                                       )

    # remove wrong chorales:
    min_pitches, max_pitches = compute_min_max_pitches(chorale_list,
                                                       voices=voice_ids)

    make_transposition_dataset(chorale_list, pickled_dataset,
                               voice_ids=voice_ids,
                               metadatas=metadatas)


def make_transposition_dataset(chorale_list, dataset_name,
                               voice_ids=[SOP_INDEX], metadatas=None):
    X = []
    index2notes, note2indexes = create_index_dicts(chorale_list,
                                                   voice_ids=voice_ids)

    # todo clean this part
    min_max_midi_pitches = np.array(
        list(map(lambda d: _min_max_midi_pitch(d.values()), index2notes)))
    min_midi_pitches = min_max_midi_pitches[:, 0]
    max_midi_pitches = min_max_midi_pitches[:, 1]
    for chorale_file in tqdm(chorale_list):
        try:
            chorale = converter.parse(chorale_file)

            midi_pitches = [
                [n.pitch.midi for n in chorale.parts[voice_id].flat.notes] for
                voice_id in voice_ids]
            min_midi_pitches_current = np.array([min(l) for l in midi_pitches])
            max_midi_pitches_current = np.array([max(l) for l in midi_pitches])
            min_transposition = max(
                min_midi_pitches - min_midi_pitches_current)
            max_transposition = min(
                max_midi_pitches - max_midi_pitches_current)
            all_transpositions = []
            for semi_tone in range(min_transposition, max_transposition + 1):
                try:
                    # necessary, won't transpose correctly otherwise
                    interval_type, interval_nature = interval.convertSemitoneToSpecifierGeneric(
                        semi_tone)
                    transposition_interval = interval.Interval(
                        str(interval_nature) + interval_type)
                    chorale_tranposed = chorale.transpose(
                        transposition_interval)
                    inputs = chorale_to_inputs(chorale_tranposed,
                                               voice_ids=voice_ids,
                                               index2notes=index2notes,
                                               note2indexes=note2indexes
                                               )
                    md = []
                    if metadatas:
                        for metadata in metadatas:
                            # todo add this
                            if metadata.is_global:
                                pass
                            else:
                                md.append(metadata.evaluate(chorale_tranposed))
                    all_transpositions.append((inputs, md, semi_tone))
                except KeyError:
                    print('KeyError: File ' + chorale_file + ' skipped')
                except FloatingKeyException:
                    print(
                        'FloatingKeyException: File ' + chorale_file + ' skipped')
            if all_transpositions:
                X.append(all_transpositions)

        except (AttributeError, IndexError):
            pass

    dataset = (X, voice_ids, index2notes, note2indexes, metadatas)
    pickle.dump(dataset, open(dataset_name, 'wb'), pickle.HIGHEST_PROTOCOL)
    print(str(len(X)) + ' files written in ' + dataset_name)


def export_dataset(chorale_list, voice_ids=[SOP_INDEX], file_path=None):
    X = []
    index2notes, note2indexes = create_index_dicts(chorale_list,
                                                   voice_ids=voice_ids)

    min_max_midi_pitches = np.array(
        list(map(lambda d: _min_max_midi_pitch(d.values()), index2notes)))
    min_midi_pitches = min_max_midi_pitches[:, 0]
    max_midi_pitches = min_max_midi_pitches[:, 1]
    with open(file_path, 'w') as f:
        for chorale_file in tqdm(chorale_list):
            try:
                chorale = converter.parse(chorale_file)

                midi_pitches = [
                    [n.pitch.midi for n in chorale.parts[voice_id].flat.notes]
                    for voice_id in voice_ids]
                min_midi_pitches_current = np.array(
                    [min(l) for l in midi_pitches])
                max_midi_pitches_current = np.array(
                    [max(l) for l in midi_pitches])
                min_transposition = max(
                    min_midi_pitches - min_midi_pitches_current)
                max_transposition = min(
                    max_midi_pitches - max_midi_pitches_current)
                all_transpositions = []
                for semi_tone in range(min_transposition,
                                       max_transposition + 1):
                    try:
                        # necessary, won't transpose correctly otherwise
                        interval_type, interval_nature = interval.convertSemitoneToSpecifierGeneric(
                            semi_tone)
                        transposition_interval = interval.Interval(
                            str(interval_nature) + interval_type)
                        chorale_tranposed = chorale.transpose(
                            transposition_interval)
                        inputs = chorale_to_inputs(chorale_tranposed,
                                                   voice_ids=voice_ids,
                                                   index2notes=index2notes,
                                                   note2indexes=note2indexes
                                                   )
                        f.write(' '.join(list(
                            map(lambda x: index2notes[SOP_INDEX][x],
                                inputs[0]))))
                        f.write('\n')
                    except IndexError:
                        pass

            except (AttributeError, IndexError):
                pass


def indexed_seq_to_score(seq, index2note, note2index):
    """

    :param note2index:
    :param index2note:
    :param seq: voice major

    :return:
    """
    num_pitches = len(index2note)
    slur_index = note2index[SLUR_SYMBOL]

    score = stream.Score()
    voice_index = SOP_INDEX
    part = stream.Part(id='part' + str(voice_index))
    dur = 0
    f = note.Rest()
    for k, n in enumerate(seq):
        # if it is a played note
        if not n == slur_index:
            # add previous note
            if dur > 0:
                f.duration = duration.Duration(dur / SUBDIVISION)
                part.append(f)

            dur = 1
            f = standard_note(index2note[n])
        else:
            dur += 1
    # add last note
    f.duration = duration.Duration(dur / SUBDIVISION)
    part.append(f)
    score.insert(part)
    return score
