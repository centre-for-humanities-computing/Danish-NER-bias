# define function to find overlap
def find_overlap(lst1, lst2):
    lst1 = set(lst1)
    lst2 = set(lst2)

    overlap = list(lst1.intersection(lst2))
    len_overlap = len(overlap)

    return len_overlap, overlap

## function to remove duplicates ##
def remove_duplicates(all_names, names_to_filter_away):
    all_names = [name for name in all_names if name not in names_to_filter_away]
    return all_names