import pickle
import random


class Attributes():
    """Class that stores and handles the attribtues. Reduced and changed sentences in comparison to the first version."""

    def __init__(self):
        """Initialize all the attribtues in their semantic group."""

        self.quantity = ["group"]  # SHould "solitary" be moved to here?!
        self.size = ["big", "small"]
        self.movement_verbs = ["flys", "hops", "swims", "tunnels", "walks"]
        self.movement_adj = ["fast", "slow"]
        self.color = ["black", "white", "blue", "brown", "gray", "orange",
                      "red", "yellow"]  # Color
        self.activity = ["active", "inactive", "nocturnal"]
        self.shape_adj = ["furry", "hairless", "toughskin", "bulbous", "lean",  # smelly added
                          "bipedal", "quadrapedal", "weak", "strong", "smelly"]  # furry=pelzig fierce=wild, timid=schüchtern, solitary=einsam
        self.shape_nouns = ["flippers", "hands", "hooves", "pads", "paws", "longleg", "longneck", "patches", "spots", "stripes",
                            "tail", "chewteeth", "meatteeth", "buckteeth", "strainteeth", "horns", "claws", "tusks", "agility", "muscle"]
        # forager=Sammler, grazer=Graser (weidetier), scavenger=Assfresser
        self.food = ["fish", "meat", "plankton", "vegetation", "insects"]
        self.eater_description = ["forager", "grazer", "scavenger", "skimmer", "stalker", "hunter"]
        self.places = ["arctic", "coastal", "desert", "bush", "plains", "forest", "fields",
                       "jungle", "mountains", "ocean", "ground", "water", "tree", "cave", "newworld", "oldworld"]
        self.behaviour = ["fierce", "timid", "smart", "solitary", "domestic"]
        # no, oldworld: Asia, Africa, Europe, newworld: America, North America, South America, Central America
        self.characteristics = ["hibernate", "nestspot"]  # hibernate=überwintern
        self.extra = ["bipedal", "lean", "weak"]

        self.opposite = {"group": ["single", "alone"],  # quantity
                         "big": ["small", "tiny"],  # size
                         "small": ["big", "gian"],
                         "flys": ["tunnels", "dives"],  # movement_verbs
                         "hops": ["tunnels", "walks"],
                         "swims": ["tunnels", "walks"],
                         "tunnels": ["flys", "walks"],
                         "walks": ["hops", "tunnels"],
                         "fast": ["slow", "moderate"],  # movement_adj
                         "slow": ["fast", "quick"],
                         "black": ["purple", "pink"],  # color
                         "white": ["purple", "pink"],
                         "blue": ["purple", "pink"],
                         "brown": ["purple", "pink"],
                         "gray": ["purple", "pink"],
                         "orange": ["purple", "pink"],
                         "red": ["purple", "pink"],
                         "yellow": ["purple", "pink"],
                         "active": ["inactive", "sluggish"],  # activity
                         "inactive": ["active", "energetic"],
                         "nocturnal": ["diurnal", "daytime"],
                         "furry": ["bald", "hairless"],  # shape_adj
                         "hairless": ["furry", "hairy"],
                         "toughskin": ["weakskin", "thin skin"],
                         "bulbous": ["skinny", "slim"],
                         "lean": ["fat", "bulbous"],
                         "bipedal": ["tripedal", "quadrupedal"],
                         "quadrapedal": ["tripedal", "bipedal"],
                         "weak": ["strong", "muscular"],
                         "strong": ["weak", "fragile"],
                         "smelly": ["fragrant", "aromatic"],
                         "flippers": ["hands", "wings"],   # shape_nouns
                         "hands": ["flippers", "hooves"],
                         "hooves": ["hands", "wings"],
                         "pads": ["flippers", "wings"],
                         "paws": ["hooves", "wings"],
                         "longleg": ["shortleg", "tiny leg"],
                         "longneck": ["short neck", "tiny neck"],
                         "patches": ["stripes", "plain"],
                         "spots": ["stripes", "plain"],
                         "stripes": ["patches", "spots"],
                         "tail": ["flat butt", "plain back"],
                         "chewteeth": ["teethless", "strainteeth"],
                         "meatteeth": ["teethless", "buckteeth"],  # buckteeth = Hasenzähne
                         "buckteeth": ["teethless", "meatteeth"],
                         "strainteeth": ["teethless", "chewteeth"],
                         "horns": ["wings", "hands"],
                         "claws": ["hooves", "flippers"],
                         "tusks": ["teethless", "buckteeth"],
                         "agility": ["slow", "immobile"],
                         "muscle": ["skinny", "thin"],
                         "fish": ["insects", "vegetation"],  # food
                         "meat": ["insects", "plankton"],
                         "plankton": ["meat", "insects"],
                         "vegetation": ["plankton", "insects"],
                         "insects": ["plankton", "fish"],
                         "forager": ["grazer", "hunter"],  # eater_description
                         "grazer": ["hunter", "skimmer"],
                         "scavenger": ["skimemr", "forager"],
                         "skimmer": ["scavenger", "grazer"],
                         "stalker": ["forager", "grazer"],
                         "hunter": ["forager", "grazer"],
                         "arctic": ["desert", "forest"],  # places
                         "coastal": ["dester", "fields"],
                         "desert": ["arctic", "fields"],
                         "bush": ["arcitic", "coastal"],
                         "plains": ["desert", "mountains"],
                         "forest": ["desert", "coastal"],
                         "fields": ["mountains", "arctic"],
                         "jungle": ["arctic", "desert"],
                         "mountains": ["plains", "fields"],
                         "ocean": ["fields", "forest"],
                         "ground": ["ocean", "air"],
                         "water": ["ground", "air"],
                         "tree": ["plains", "fields"],
                         "cave": ["plains", "water"],
                         "newworld": ["oldworld", "arctic"],
                         "oldworld": ["newworld", "america"],
                         "fierce": ["gentle", "mild"],  # behaviour
                         "timid": ["fierce", "aggressive"],
                         "smart": ["dumb", "unwise"],
                         "solitary": ["groups", "accompanied"],
                         "domestic": ["wild", "feral"],
                         "hibernate": ["continuance", "continuation"],
                         "nestspot": ["born", "mammal"]}

        self.quantity_template = [lambda c: f'a photo of a {c} of animals.']

        self.size_template = [lambda c: f'a photo of a {c} animal.']

        self.movement_verbs_template = [lambda c: f'a photo of an animal that {c}.']

        self.movement_adj_template = [lambda c: f'a photo of a {c} animal.']

        self.color_template = [lambda c: f'a photo of a {c} animal.']

        self.activity_template = [lambda c: f'a photo of a {c} animal.']

        self.shape_adj_template = [lambda c: f'a photo of a {c} animal.']

        self.shape_nouns_template = [lambda c: f'a photo of an animal with {c}.']

        self.food_template = [lambda c: f'a photo of an animal that eats {c}.']

        self.eater_description_template = [lambda c: f'a photo of a {c} animal.']

        self.places_template = [lambda c: f'a photo of a {c} animal.']

        self.behaviour_template = [lambda c: f'a photo of a {c} animal.']

        self.characteristics_template = [lambda c: f'a photo of a {c} animal.']

        self.attribute_dict = {"quantity": (self.quantity, self.quantity_template),
                               "size": (self.size, self.size_template),
                               "movement_verbs": (self.movement_verbs, self.movement_verbs_template),
                               "movement_adj": (self.movement_adj, self.movement_adj_template),
                               "color": (self.color, self.color_template),
                               "activity": (self.activity, self.activity_template),
                               "shape_adj": (self.shape_adj, self.shape_adj_template),
                               "shape_nouns": (self.shape_nouns, self.shape_nouns_template),
                               "food": (self.food, self.food_template),
                               "eater_description": (self.eater_description, self.eater_description_template),
                               "places": (self.places, self.places_template),
                               "behaviour": (self.behaviour, self.behaviour_template),
                               "characteristics": (self.characteristics, self.characteristics_template),
                               "extra": (self.extra, None)}

    def attribute_to_text(self, attribute):
        """Return the attribute in a list of sentences that fit the attribtue."""

        for key_attr in self.attribute_dict.keys():
            if key_attr == "extra":
                continue
            attr_list = self.attribute_dict[key_attr][0]
            if attribute in attr_list:
                # template = self.attribute_dict[key_attr][1]
                template = [
                    lambda c: f"a photo of a {c} animal.",
                    lambda c: f"a picture of a {c} animal.",
                    lambda c: f"a photograph of a {c} animal."]
                texts = [templ(attribute) for templ in template]
                return texts
        print("Invalid attribute! Attribute not found.")
        return None

    def attribute_to_tuple_opposite(self, attribute):
        """Return the attribute in two lists of sentences (positive, negated) that fit the attribute."""

        for key_attr in self.attribute_dict.keys():
            if key_attr == "extra":
                continue
            attr_list = self.attribute_dict[key_attr][0]
            if attribute in attr_list:
                # template = self.attribute_dict[key_attr][1]
                template_pos = [
                    lambda c: f"a photo of a {c} animal.",
                    lambda c: f"a picture of a {c} animal.",
                    lambda c: f"a photograph of a {c} animal."]
                template_neg = [
                    lambda c: f"not a photo of a {c} animal.",
                    lambda c: f"not a picture of a {c} animal.",
                    lambda c: f"not a photograph of a {c} animal."]
                texts_pos = [templ(attribute) for templ in template_pos]
                texts_neg = [templ(attribute) for templ in template_neg]
                return texts_pos, texts_neg
        print("Invalid attribute! Attribute not found.")
        return None

    def attribute_to_tuple(self, attribute):
        """Return the attribute in two lists of sentences (positive, negated) that fit the attribute."""

        for key_attr in self.attribute_dict.keys():
            if key_attr == "extra":
                continue
            attr_list = self.attribute_dict[key_attr][0]
            if attribute in attr_list:
                # template = self.attribute_dict[key_attr][1]
                template = [
                    lambda c: f"a photo of a {c} animal.",
                    lambda c: f"a picture of a {c} animal.",
                    lambda c: f"a photograph of a {c} animal."]
                texts_pos = [templ(attribute) for templ in template]
                neg_attr = random.choice(self.opposite[attribute])
                texts_neg = [templ(neg_attr) for templ in template]
                return texts_pos, texts_neg
        print("Invalid attribute! Attribute not found.")
        return None

    def keep_category(self, category: str, idx_attribute: dict, leave_out=[]):
        """Removes every attribute from idx_attribute that is not in the specified category."""

        if category in self.attribute_dict.keys():
            rem_keys = []
            attr_list = set(self.attribute_dict[category][0])
            attr_list = attr_list.difference(set(leave_out))
            for key in idx_attribute:
                if idx_attribute[key] not in attr_list:
                    rem_keys.append(key)
            for key in rem_keys:
                del idx_attribute[key]
            return idx_attribute
        else:
            print("Not a valid category!")
            return None

    def keep_good_attributes(self, attr_idx_label_mapping, cut_off=0.5, file_path="/home/felix/new_bachelor/awa2/results/predict_attribute/opposite4_attr_relative_results.pickle"):
        """Only keep attributes with recall over a cut off value. Returns the removed labels."""

        with open(file_path, "rb") as act:
            relative_results = pickle.load(act)
        act.close()
        rec = [relative_results[key]["true_positive"] /
               (relative_results[key]["true_positive"] + relative_results[key]["false_negative"]) for key in range(1, 86)]
        labels = [attr_idx_label_mapping[key] for key in range(1, 86)]
        rem_attr = []
        for perf, name in zip(rec, labels):
            if perf < cut_off:
                rem_attr.append(name)
        return rem_attr
