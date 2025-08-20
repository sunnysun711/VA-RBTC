import json


class Train:
    r"""
    A class to represent a train with specific type and load characteristics.

    Initialize the Train instance with the specified type, loaded standing passenger density, and weight per person.
    Load and combine train-specific and public train data from JSON files.

    Parameters
    ----------
    type_ : str
        The type of the train, used to load specific data from JSON files.
    load_spd : float, optional
        The loaded standing passenger density (number of passengers per square meter). Default is 0.
    weight_per_person_kg : float, optional
        The average weight per person in kilograms. Default is 68 kg.

    Attributes
    ----------
    type : str
        The type of the train.
    data : dict
        A dictionary containing train-specific data combined with public train data.
    load_spd : float
        The loaded standing passenger density (number of passengers per square meter).
    weight_per_person_kg : float
        The average weight per person in kilograms.

    Examples
    --------
    Initialize a Train instance with type "Wu2021Train"

    >>> train = Train(type_="Wu2021Train")

    Initialize a Train instance with type "Wu2021Train", a loaded standing passenger density of 6 passengers per square
    meter, and an average weight per person of 70 kg

    >>> train = Train(type_="Wu2021Train", load_spd=6, weight_per_person_kg=70)
    """

    def __init__(self, type_: str, load_spd: float = 0, weight_per_person_kg: float = 68):
        """
        Constructs all the necessary attributes for the Train object.

        Parameters
        ----------
        type_ : str
            The type of the train, used to load specific data from JSON files.
        load_spd : float, optional
            The loaded standing passenger density (number of passengers per square meter). Default is 0.
        weight_per_person_kg : float, optional
            The average weight per person in kilograms. Default is 68 kg.
        """
        self.type: str = type_
        data_public = json.load(open("data/train/TrainPublic.json", "r", encoding="utf-8"))
        self.data = json.load(open(f'data/train/{type_}.json', "r", encoding="utf-8"))
        self.data.update(data_public)
        self.data['mass'] += (load_spd * self.data['area'] * weight_per_person_kg) / 1000  # from kg to t
        self.data['load_spd'] = load_spd
        self.data['weight_per_person_kg'] = weight_per_person_kg


def main():
    t = Train("Scheepmaker2020", load_spd=3)
    print(t.data)
    pass


if __name__ == '__main__':
    main()
