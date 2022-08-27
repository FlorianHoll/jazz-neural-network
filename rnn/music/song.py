"""A song in the training corpus."""
from bs4 import BeautifulSoup


class Song:
    """A song that will be used to train the neural network."""

    def __init__(self, filename: str):
        """Initialize the song."""
        with open(filename, "r") as file:
            data = file.read()

        data = BeautifulSoup(data, "xml")
        print(data.prettify())

        self._xml_representation = data
        self.music_representation = None
        self.neural_net_representation = None

    def parse(self):
        """Parse the song from the xml representation."""
        pass

    def transpose(self, steps: int):
        """Transpose the whole song by x half tones."""
        pass
