# Jazz Neural Network
A neural net that is able to write jazz standards in the style of the Real Book.

## General information
The project consists of two neural networks that work together to write jazz songs
in the style of the Real Book:
- A neural network that writes the harmony, i.e. the chords
- A neural network that writes some melody over the harmony

The networks are trained on the Real Book, a corpus of songs that comprise well-known
Jazz songs, so called "Standards".

When predicting, one can pass either an actual Real Book song (either a specific one or
a random one) or a self-composed snippet as the input. First, the harmony network will
write some harmony for the amount of measures that are passed as a parameter. Then, the
melody network will write some melody that fits the chords. For both prediction
processes, BeamSearch is used (see below). Finally, the predictions are combined into
a .xml file that can be read by using music notation software.

## Problem/ Modeling approach
Both networks are Recurrent Neural Networks that aim to predict the next note/
chord, given previous notes/ chords.

In order to achieve this, the respective network is given sequences of notes/ chords
of a fixed length as input and has to predict the next note/ chord as a
**classification problem** where each note/ chord is one "category". The network has
to learn which of the 88 notes or 48 chords (see below for explanation) it has to
choose. The loss that is used is Categorical Crossentropy Loss.

When predicting, the network is fed the end of a song (the so-called "Turnaround") and
then has to predict the next note/ chord, i.e. the beginning of the song.
It is then iteratively fed with its own predictions until the specified number of
measures is written.

As already stated, the harmony is written first, followed by the melody which is written
"on top" of the harmony. The final song is then converted to a format that can be read
by music notation software.

## Data Preprocessing

### 1.) Transformation of notes into NN-compatible format

To transform music into a problem that a neural network can solve, some preprocessing
is necessary: Each song first needs to be converted a numerical representation that
the network can handle. For this purpose, the classes `Note` and `Chord` classes from
the `rnn.music.musical_elements` module are used.

Both the `Note` and `Chord` classes have three main properties:
* The **pitch height**, represented in MIDI-format. The MIDI-format is a useful tool
  here because it creates a mapping between the frequency of a note (i.e. its pitch
  height) and an integer between 21 and 108, such that a half-tone step increase in
  pitch height corresponds to an increase of the MIDI number of 1. This means that
  the note *C4* can be represented as *60* in MIDI format, the note C#4=61, D4=62, C5=72
  etc.
* The **duration** of the element: The duration of the note is represented as an integer
  between 1 and 48, where 48 corresponds to a whole tone (four beats), 24 = half tone
  (two beats), 12 = quarter tone (one beat) etc. This representation was chosen because
  everything down to sixteenth triplets can be represented as an integer. These integers
  later corresponds to neural network output nodes--therefore, it is convenient to
  have the durations already in integer format.
* The **offset** of the element within the measure: The offset is represented in the
  same way as the duration; however, it marks the beginning of the note within the
  measure. This means that if a note starts right at the beginning of a measure, its
  offset is 0; if it starts one beat, i.e. a quarter note after the measures start,
  its offset is 12; if it starts two beats after the start, its offset is 24 etc.
  Note that the offset can only take values between 0 and 47 (48 is not possible
  because a note cannot have a duration of 0).

The `Note` class represents a musical note with the described properties and comes
with some additional functionality. Its transformation works in both ways (i.e. from
note to MIDI pitch height as well as the other way around). In order to do this, it can
be instantiated from different representations with the `Note._from_pitch_height()` and
`Note.from_symbol()` methods. The duration and offset are properties of the note that
do not change regardless of the pitch representation.

Since chords consist of several notes put together, the `Chord` class works the same
way: A `Chord` object consists of several `Note` objects, all with their respective
properties. It can also be transformed in both ways and instantiated from different
representations in the exact same way as the `Note` class.

### 2.) Training input format
The notes from a Real Book song are parsed and transformed by the above classes; they
are then formatted by using a "sliding window" approach: Each note/ chord of the song
is used as a target once. All the N notes before are used as the input sequence (N is
the sequence length; it is fixed before training the network) and the note itself is
used as the target. This is done for each note of the song.

### 3.) Data Augmentation
There are 12 keys that a song can be written in. However, some keys are more common in
Jazz than others. In order to avoid that only certain keys are learnt sufficiently and
to augment the training data, each song is transposed to each of the 12 keys. This means
that the amount of training data is twelve times the original data. Furthermore, this
ensures that the network will later be able to write songs in each key.

## Model architecture
The models are both Recurrent Neural Networks. Specifically, the architecture is as
follows:
