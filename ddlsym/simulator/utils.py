def _type_error(expected, got):
    return TypeError("Expected a {} as argument but got {} .".format(expected, type(got)))
