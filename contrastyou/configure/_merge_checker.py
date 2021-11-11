import difflib

from .dictionary_utils import mapType


def __name_getter(dictionary: mapType, previous_name, previous_names):
    for k, v in dictionary.items():
        if previous_name == "":
            previous_names.append(k)
        else:
            previous_names.append(str(previous_name) + "." + str(k))
    for k, v in dictionary.items():
        if isinstance(v, mapType):
            __name_getter(v, str(k) if previous_name == "" else str(previous_name) + "." + str(k), previous_names, )


def merge_checker(base_dictionary, coming_dictionary):
    base_names, coming_names = [], []
    __name_getter(base_dictionary, "", base_names), __name_getter(coming_dictionary, "", coming_names)

    undesired_attributes = sorted(set(coming_names) - set(base_names))

    def create_possible_suggestion(unwanted_string: str):
        candidate_list = difflib.get_close_matches(unwanted_string, base_names, n=1)
        if len(candidate_list) > 0:
            return candidate_list[0]
        else:
            return ""

    if len(undesired_attributes) > 0:
        raise RuntimeError(
            f"\nUnwanted attributed identified compared with base config: \t"
            f"{', '.join([f'`{x}`: (possibly `{create_possible_suggestion(x)}`)' for x in undesired_attributes])}"
        )


if __name__ == "__main__":
    base = {1: {"a": 1, "b": 2}, 2: ["C", "D"]}
    inc_dict = {1: {"a": "replace", "ef": 2}}
    merge_checker(base, coming_dictionary=inc_dict)
