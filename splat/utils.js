function floatToHalf(float) {
    _floatView[0] = float;
    var f = _int32View[0];

    var sign = (f >> 31) & 0x0001;
    var exp = (f >> 23) & 0x00ff;
    var frac = f & 0x007fffff;

    var newExp;
    if (exp == 0) {
        newExp = 0;
    } else if (exp < 113) {
        newExp = 0;
        frac |= 0x00800000;
        frac = frac >> (113 - exp);
        if (frac & 0x01000000) {
            newExp = 1;
            frac = 0;
        }
    } else if (exp < 142) {
        newExp = exp - 112;
    } else {
        newExp = 31;
        frac = 0;
    }

    return (sign << 15) | (newExp << 10) | (frac >> 13);
}

export {floatToHalf};