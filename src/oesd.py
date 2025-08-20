# this file get oesd
import json


class OESD:
    """
    A class to represent an Onboard Energy Storage Device (OESD) with specific type.

    Initialize the OESD instance with the specified type.
    Load and combine OESD-specific and public OESD data from JSON files.

    Parameters
    ----------
    type_ : str
        The type of the OESD. Should be either "Li-ion", "supercapacitor", or "flywheel".

    Attributes
    ----------
    type : str
        The type of the OESD.
    data : dict
        A dictionary containing OESD-specific data combined with public OESD data.

    Examples
    --------
    Initialize an OESD instance with type "Li-ion"

    >>> oesd = OESD(type_="Li-ion")
    >>> oesd.data
    {'charge': {'x': [0, 0.7, 0.9, 1], 'y': [80000, 49200, 23950, 0]}, 'discharge': {'x': [0, 0.15, 0.4, 1],
    'y': [0, 26520, 49580, 80000]}, 'mass': 0.08, 'capacity': 13.88, 'investment': 150, 'comment': 'investment
    表示电池投资（单位k$）。charge和discharge表示线性近似分段点，x的单位是100%，y的单位是kW。mass表示电池重量，单位t。capacity
    表示电池容量，单位kWh'}


    Initialize an OESD instance with a type that is not found in the data files

    >>> oesd = OESD(type_="unknown")
    OESD 'unknown' not found!
    """

    def __init__(self, type_: str):
        """
        Initialize the OESD object.

        Parameters
        ----------
        type_ : str
            The type of the OESD. Should be either "Li-ion", "supercapacitor", or "flywheel".
        """
        self.type = type_
        data_public: dict = json.load(open(f"data/OESD/OESDPublic.json", "r", encoding="utf-8"))
        try:
            oesd_data = json.load(open(f"data/OESD/{type_}.json", 'r', encoding="utf-8"))
            self.data: dict = oesd_data
            self.data.update(data_public)
        except FileNotFoundError:
            print(f"OESD '{type_}' not found!")
            self.data: dict = {"mass": 0}
            self.type = None


def main():
    sc = OESD("supercapacitor")
    print(sc.data)
    pass


if __name__ == '__main__':
    main()
