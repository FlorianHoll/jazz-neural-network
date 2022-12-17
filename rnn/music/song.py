"""A song in the training corpus."""
import logging
import re
from typing import List
from typing import Union

import bs4
import numpy as np
from bs4 import BeautifulSoup

from rnn.music.musical_elements import Chord
from rnn.music.musical_elements import Note
from rnn.music.musical_elements import RestChord
from rnn.music.musical_elements import RestNote
from rnn.music.utils import chord_type_to_compatible_chord
from rnn.music.utils import sharps_to_key_signature_symbol

logging_formatter = logging.Formatter(
    "%(asctime)s|%(levelname)-8s|%(filename)-25s|%(lineno)-4s|%(message)s"
)
handler = logging.StreamHandler()
handler.setFormatter(logging_formatter)

# Add handler and set level.
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class SongParser:
    """Parse one song and convert to a neural network-compatible input.

    The songs are in the .xml format, therefore BeautifulSoup is used to
    parse the basic structure. We are interested in notes and chord symbols;
    therefore, only these types of information will be parsed. The SongParser
    parses the measures one at a time and returns a list of the musical
    elements that constitute the song.
    Afterwards, this parsed song can be converted into an input to the neural
    net. For this, the structure will be converted into multidimensional
    arrays consisting only of a numerical representation of the musical
    elements.

    The SongParser is the base class. There are two neural networks, one
    for writing the harmony of the song and one for writing the melody.
    Therefore, two child classes for these specific networks exist because
    the networks need different types of input (described in detail in
    the respective docstring of the child class).

    :param filename: The name of the .xml file to be parsed.
    """

    def __init__(self, filename: str):
        """Initialize the song."""
        logger.info(
            f"Instantiating parser for song "
            f"{re.sub('(data*.)|([./])|(xml*)', '', filename)}"
        )
        if filename[-4:] != ".xml":
            raise TypeError(
                f"The input file {filename} is in the wrong format; "
                "only .xml files can be parsed."
            )
        with open(filename, "r") as file:
            data = file.read()

        self.raw_data = BeautifulSoup(data, "xml")

        # Since the .xml files differ in their representaion
        #   of the duration, we need a duration multiplier.
        #   This is consistent across the whole song; therefore,
        #   it can be represented as an attribute of the song.
        self.duration_multiplier = None

        # Initialize empty structures for the musical elements.
        #   The lists will be filled as the measures are parsed.
        self.melody_representation = []
        self.harmony_representation = []

    @property
    def key_signature(self) -> tuple[str]:
        """Get the key signature of the whole song.

        :return: The key signature as a symbol, e.g. "C" or "F#".
        """
        key_signature_in_fifths_from_c = int(
            self.raw_data.find("key").find("fifths").string
        )
        return sharps_to_key_signature_symbol(key_signature_in_fifths_from_c)

    def parse(self):
        """
        Parse the song from the xml representation.

        This means iteratively parsing all measures. For each measure,
        the musical elements that the measure consists of will be
        parsed with the help of the musical elements classes. In the
        end, we have a list that contains the representation of the
        musical elements that the song consists of.
        """
        measures = self.raw_data.findChildren("measure")
        self._find_duration_multiplier(measures[1])
        for measure in measures:
            self._parse_one_measure(measure)

    def _find_duration_multiplier(self, measure: bs4.element.Tag) -> None:
        """Find the duration multiplier for the whole song.

        The duration representation unfortunately differs across some of .yml
        files. Therefore, we calculate the overall duration of the second
        measure (since the first measure can be an upbeat) and divide by 48
        - the standardized duration of one measure - to obtain a multiplier.
        If, for example, the durations of one measure add up to 12, then all
        durations have to be multiplied by 4 to obtain a sensible representation.
        Therefore, 4 is the multiplier value.

        :param measure: The second measure of the song, represented in the format
            parsed by bs4.
        """
        all_durations_in_measure = [dur.string for dur in measure.findAll("duration")]
        summed_duration_of_measure = (
            np.array(all_durations_in_measure).astype(int).sum()
        )
        self.duration_multiplier = 48 / summed_duration_of_measure
        logger.debug(
            f"The summed duration of measure 1 is {summed_duration_of_measure}; "
            f"therefore, the multiplier is {self.duration_multiplier}."
        )

    def _parse_one_measure(self, measure: bs4.element.Tag) -> None:
        """Parse one measure and convert the elements to musical elements.

        The .xml file consists of the notes and chords; however, in a format
        that is hard to work with. Therefore, we need to convert them into
        the musical elements classes to easily work with them.
        The actual implementation is handled by the child classes.
        """
        pass

    def _parse_one_note(
        self, note: bs4.element.Tag, offset: int
    ) -> Union[Note, RestNote]:
        """Parse one note and return it.

        We obtain the information about the note height and its duration.
        The offset was counted and does not need to be parsed. With this
        information, we create a Note element.

        :param note: The note to be parsed (as a BeautifulSoup representation
            of the .xml element).
        :param offset: The offset that was updated throughout the measure.
        :return: The note to be added to the melody representation; either a
            Note object or a RestNote object if the note was a rest symbol.
        """
        note_duration = int(note.find("duration").string)
        note_duration *= self.duration_multiplier  # correct for individual parsings

        # If the note is NOT a rest, it will have the 'step'
        #   and 'octave' attributes; however, if it IS a rest,
        #   an AttributeError will be raised because the attribute
        #   does not exist.
        try:
            note_symbol = note.find("step").string
            note_octave = note.find("octave").string
            try:
                # If the note is altered (i.e. raised or flattened),
                #   this information has to be added to the note symbol.
                note_alteration = int(note.find("alter").string)
                if note_alteration == -1:
                    note_symbol += "b"
                elif note_alteration == 1:
                    note_symbol += "#"
            except AttributeError:
                pass
            final_note_symbol = f"{note_symbol}{note_octave}"
            note_to_add = Note.from_symbol(final_note_symbol, note_duration, offset)

        # If an AttributeError occurred while trying to obtain the symbol, we know
        #   that the element is a rest. The duration and offset apply nonetheless.
        except AttributeError:
            note_to_add = RestNote(note_duration, offset)

        return note_to_add

    def _parse_one_harmony_symbol(
        self, harmony: bs4.element.Tag, offset: int
    ) -> Union[RestChord, Chord]:
        """Parse one harmony symbol and return it.

        We obtain the information about the note height and its duration.
        The offset was counted and does not need to be parsed. With this
        information, we create a Note element.

        :param harmony: The chord to be parsed (as a BeautifulSoup representation
            of the .xml element).
        :param offset: The offset that was updated throughout the measure.
        :return: The chord to be added to the harmony representation; either a
            Chord, an AccompanyingChord, or a RestChord object.
        """
        root = harmony.find("root-step").string
        chord_type = harmony.find("kind").get("text")  # get the extension symbol
        # Change representation depending on what network the chord
        #   element will be used for.

        # If the chord has an offset, use it - this indicates that there is only
        #   one note with several chords played over it. Only in this case will
        #   will the .xml representation have this attribute. In this case, using
        #   the durations of the notes to update the offset obviously does not work.
        new_offset = harmony.find("offset")
        if new_offset is not None:
            offset += int(new_offset.string)

        if chord_type == "N.C.":
            chord_to_add = RestChord(offset=offset)
        else:
            try:
                # If the chord is altered (i.e. raised or flattened),
                #   this information has to be added to the root symbol.
                note_alteration = int(harmony.find("root-alter").string)
                if note_alteration == -1:
                    root += "b"
                elif note_alteration == 1:
                    root += "#"
            except AttributeError:
                pass

            # In some rare cases, the harmony is represented differently in the
            #   .xml file. In these cases, we will use the "kind" attribute.
            if chord_type is None:
                chord_type = harmony.find("kind").string
            chord_type = self._convert_to_compatible_chord_type(chord_type)

            final_chord_symbol = f"{root} {chord_type}"
            chord_to_add = Chord.from_symbol(final_chord_symbol, offset=int(offset))

        return chord_to_add

    @staticmethod
    def _convert_to_compatible_chord_type(chord_type: str) -> str:
        """Convert an arbitrary chord type to a Chord-compatible type.

        In the .xml files, there are many different representations
        of chord types. For example, a minor7 chord can be represented
        as "m7", "min7", "-7", m", "-" etc.; such a chord type will be
        converted to simply "min7" (which the chord class works with).

        :param chord_type: The chord type given by the .xml representation.
        """
        return chord_type_to_compatible_chord(chord_type)

    def _convert_and_augment_training_data(
        self, input_length: int, target_length: int
    ) -> np.ndarray:
        """Convert and augment training data.

        The implementation will be handled by the children classes.

        :param input_length: The input length of the sequences for the neural net.
        :param target_length: The length of the target to be predicted by the
            neural net.
        :return: The augmented, i.e. transposed and slided, data, with all information
            that the neural net needs.
        """
        pass

    def _transpose_and_slide(
        self,
        musical_elements: Union[List[Chord], List[Note]],
        attribute: str,
        input_length: int,
        target_length: int,
    ) -> np.ndarray:
        """Augment the data by transposing to each key and applying a sliding window.

        This means that the neural net will receive the training
        data in all keys to (1.) avoid over-representation of
        some key signatures (some key signatures are more common
        than others in jazz) and to (2.) lead to generalization:
        The network should understand that the relations between
        notes and chords are the same for all keys alike.

        Furthermore, if the song has N musical elements, a sliding
        window is applied to have N sequences of input_length with a
        target of length target_length at the end.

        Consider, for example, the following example song which
        is already in neural net representation (i.e. the elements
        have been converted to numbers):

        [1, 2, 3, 4, 5, 6]

        A sliding window is applied to obtain 6 sequences out of this.
        If input_length = 3 and target_length = 1, the resulting
        representation would look as follows:

        [[1, 2, 3, 4],
         [2, 3, 4, 5],
         [3, 4, 5, 6],
         [4, 5, 6, 1],
         [5, 6, 1, 2],
         [6, 1, 2, 3]]

        where [:, -1], i.e. the last column, is the target that will be
        predicted by the network and [:, :input_length] is the input to
        the network.

        This essentially means that the network will predict each note/ chord
        in the song given the last notes/ chords.

        :param musical_elements: The musical elements (either chords
            or notes) that the song consists of)
        :param attribute: The attribute of each element to use, e.g. "duration",
            "offset", or "neural_net_representation".
        :param input_length: The input length of the sequences for the neural net.
        :param target_length: The length of the target to be predicted by the
            neural net.
        :return: The augmented, i.e. transposed and slided, data.
        """
        augmented_data = None
        for transpose_steps in range(-6, 6):
            elements = np.array(
                [
                    getattr(element.transpose(transpose_steps), attribute)
                    for element in musical_elements
                ]
            )
            elements = self._sliding_window(elements, input_length, target_length)
            augmented_data = (
                elements
                if augmented_data is None
                else np.vstack([augmented_data, elements])
            )

        return augmented_data

    @staticmethod
    def _sliding_window(
        elements_to_slide: np.ndarray, sequence_length: int, target_length: int
    ) -> np.ndarray:
        """Apply a sliding window to the elements.

        The process is explained in detail in the docstring of the
        _transpose_and_slide() method.

        :param elements_to_slide: The elements that the sliding window
            shall be applied to (length N).
        :param sequence_length: The length of the input sequence (has to be positive).
        :param target_length: The length of the target (has to be positive).
        :return: An array of slided windows with shape
            N x (sequence_length + target_length).
        """
        if (sequence_length <= 0) or (target_length <= 0):
            raise ValueError("Both sequence and target length have to be > 0.")
        slided = np.array(
            [np.roll(elements_to_slide, -i) for i, _ in enumerate(elements_to_slide)]
        )
        return slided[:, : (sequence_length + target_length)]


class HarmonySongParser(SongParser):
    """Parse a song's harmony."""

    def _parse_one_measure(self, measure: bs4.element.Tag):
        """Parse the harmony of one measure.

        The elements in the measure are iterated through and the harmony
        elements are converted to Chord objects. The note objects are used
        to keep track of the offset, i.e. where we are in the measure (since
        this information is unfortunately missing in the .xml file).
        Each Chord object is then appended to the harmony representation.

        :param measure: The whole measure as parsed by bs4.
        """
        elements = measure.findChildren(["note", "harmony"])  # find all notes.
        # The offset is set to 0 initially and will be updated iteratively.
        offset = 0
        for element in elements:
            # First, find out if the element is a note or a harmony symbol.
            if element.name == "note":
                note_duration = (
                    int(element.find("duration").string) * self.duration_multiplier
                )
                # The offset must be updated to keep track of the position
                #   of the measure that is currently being parsed.
                offset += note_duration
            else:
                harmony = self._parse_one_harmony_symbol(element, offset)
                self.harmony_representation.append(harmony)
        assert offset == 48

    def _convert_and_augment_training_data(
        self, input_length: int, target_length: int
    ) -> np.ndarray:
        """Convert and augment the harmony data into the neural net format.

        For the harmony neural network, the relevant information is:
        - The chord (represented as a number)
        - Its duration
        - Its offset

        :param input_length: The input length of the sequences for the
            neural net.
        :param target_length: The length of the target to be predicted
            by the neural net.
        :return: The augmented, i.e. transposed and slided, data, with
            all information that the neural net needs.
        """
        self._calculate_chord_durations()
        attributes_to_augment = [
            "neural_net_representation",
            "duration",
            "offset",
        ]
        chord_types, durations, offsets = [
            self._transpose_and_slide(
                self.harmony_representation, attribute, input_length, target_length
            )
            for attribute in attributes_to_augment
        ]
        return np.array([chord_types, durations, offsets])

    def _calculate_chord_durations(self):
        """
        Calculate the chord durations.

        This needs to be done because the .xml files unfortunately
        do not contain information regarding the duration of the chords.
        During the parsing, their offset offset is calculated by looking
        at the duration of the notes that were played up to a certain point.
        Therefore, all the information available is the offsets, the durations
        need to be calculated from them. This is done as follows:
        The offsets are always numbers between 0 and 48, representing the
        point of the measure where the chord is hit.

        If the offsets are the following:

        [0, 12, 18, 24, 36, 0, 24, 0, 12, 36]

        Then the durations are calculatd as follows:

        1.) Shift the durations one to the right.
        [12, 18, 24, 36, 0, 24, 0, 12, 36, 0]

        2.) Subtract the offsets from the shifted offsets.
            [12,  18,  24,  36,   0,  24,   0,  12,  36,   0]
        -   [0,   12,  18,  24,  36,   0,  24,   0,  12,  36]
        =   [12,  6,    6,  12, -36,  24, -24,  12,  24, -36]

        3.) Add 48 to all resulting values <= 0:
        =   [12,  6,    6,  12,  12,  24,  24,  12,  24,  12]

        4.) Overwrite the original duration attributes of the
            Chord objects (the original durations were merely a
            placeholder).
        """
        offsets = [chord.offset for chord in self.harmony_representation]
        chord_offsets_offset_by_one = np.roll(offsets, -1)
        chord_durations = chord_offsets_offset_by_one - offsets
        chord_durations[chord_durations <= 0] += 48
        for chord, duration in zip(self.harmony_representation, chord_durations):
            chord.duration = duration

    def parse_and_return_neural_net_input(
        self, input_length: int = 8, target_length: int = 1
    ) -> np.ndarray:
        """Parse the song and return the neural net input.

        :param input_length: The input length of the sequences for the neural net.
        :param target_length: The length of the target to be predicted by the
            neural net.
        :return: The augmented, i.e. transposed and slided, data, with all information
            that the neural net needs.
        """
        self.parse()
        return self._convert_and_augment_training_data(input_length, target_length)


class MelodySongParser(SongParser):
    """Parse a song's melody."""

    def _parse_one_measure(self, measure):
        """Parse the melody of one measure.

        The elements in the measure are iterated through and the note
        elements are converted to Note objects. The harmony objects are
        converted to Chord objects to keep track of which chord was played
        over each note (this information is relevant for the harmonic
        context of the note).
        Each object is appended to the harmony and melody representation.

        :param measure: The whole measure as parsed by bs4.
        """
        elements = measure.findChildren(["note", "harmony"])
        # The offset is set to 0 initially and will be updated iteratively.
        offset = 0
        current_accompanying_harmony = (
            self.harmony_representation[-1]
            if len(self.harmony_representation) > 0
            else RestChord()
        )
        for element in elements:
            # First, find out if the element is a note or a harmony symbol.
            #   Note: Rests are also a "note" object in the .xml file.
            if element.name == "note":
                note = self._parse_one_note(element, offset)
                self.melody_representation.append(note)
                self.harmony_representation.append(current_accompanying_harmony)
                # The offset must be updated to keep track of the position
                #   of the measure that is currently being parsed.
                offset += note.duration
            else:
                current_accompanying_harmony = self._parse_one_harmony_symbol(
                    element, offset
                )
        assert offset == 48  # Assert that the measure length works out.

    def _convert_and_augment_training_data(
        self, input_length: int, target_length: int
    ) -> np.ndarray:
        """Convert and augment the harmony data into the neural net format.

        For the harmony neural network, the relevant information is:
        - The melody (represented as its MIDI pitch)
        - Its duration
        - Its offset
        - the chord played underneath, i.e. the harmonic context. This
            is represented as the four notes that the chord consists of
            (the MIDI pitches).

        :param input_length: The input length of the sequences for the
            neural net.
        :param target_length: The length of the target to be predicted
            by the neural net.
        :return: The augmented, i.e. transposed and slided, data, with
            all information that the neural net needs.
        """
        note_attributes_to_augment = ["neural_net_representation", "duration", "offset"]
        melody_notes, durations, offsets = [
            self._transpose_and_slide(
                self.melody_representation, attribute, input_length, target_length
            )
            for attribute in note_attributes_to_augment
        ]
        accompanying_chords = self._transpose_and_slide(
            self.harmony_representation,
            "pitch_neural_net_representation",
            input_length,
            target_length,
        )
        (
            chord_note_1,
            chord_note_2,
            chord_note_3,
            chord_note_4,
        ) = accompanying_chords.swapaxes(0, 2).swapaxes(1, 2)

        return np.array(
            [
                melody_notes,
                durations,
                offsets,
                chord_note_1,
                chord_note_2,
                chord_note_3,
                chord_note_4,
            ]
        )

    def parse_and_return_neural_net_input(
        self, input_length: int = 8, target_length: int = 1
    ) -> np.ndarray:
        """Parse the song and return the neural net input.

        :param input_length: The input length of the sequences for the neural net.
        :param target_length: The length of the target to be predicted by the
            neural net.
        :return: The augmented, i.e. transposed and slided, data, with all information
            that the neural net needs.
        """
        self.parse()
        return self._convert_and_augment_training_data(input_length, target_length)
