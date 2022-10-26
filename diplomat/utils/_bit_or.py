def _bit_or(*flags: int) -> int:
    """
    Combine flags into a single flag, ignoring non-integer values...

    :param flags: A list of or integers, that need to be bitwise or-ed together.

    :return: The combined integer flag...
    """
    total = 0

    for flag in flags:
        if(isinstance(flag, int)):
            total |= flag

    return total
