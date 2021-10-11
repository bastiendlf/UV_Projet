import numpy as np


def get_quantification_matrix(quality: int = 50):
    """
    Compute quantization matrix for JPEG compression
    :param quality: int between 1 and 100
    :return: numpy 2D array with quantification coefficients
    """
    q50 = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                    [12, 12, 14, 19, 26, 58, 60, 55],
                    [14, 13, 16, 24, 40, 57, 69, 56],
                    [14, 17, 22, 29, 51, 87, 80, 62],
                    [18, 22, 37, 56, 68, 109, 103, 77],
                    [24, 35, 55, 64, 81, 104, 113, 92],
                    [49, 64, 78, 87, 103, 121, 120, 101],
                    [72, 92, 95, 98, 112, 100, 130, 99]])

    if quality < 50:
        s = 5000 / quality
    else:
        s = 200 - (2 * quality)

    q_output = np.floor((s * q50 + 50) / 100)
    q_output[q_output == 0] = 1
    return np.array(q_output, dtype=np.int)


# Rounding Operators
def flt2fix(x, k):
    """ Convert from floating- to fixed-point"""
    return int(x * 2 ** k + 0.5)


def floor(x, y, b=1):
    """ Compute z = floor(x/y)"""
    if b == 1:
        z = int(x) >> int(np.log2(y))
    else:
        z = int(x / y)
    return z


def halfup(x, y, b=1):
    """Compute z = halfup(x/y)"""
    x = x + (int(y) >> 1)
    return floor(x, y, b)


def trunc(x, y, b=1):
    """Compute z = trunc(x/y)"""
    s = np.sign(x)
    x = abs(x)
    z = floor(x, y, b)
    return s * z


def round_op(x, y, b=1):
    """Compute z = round(x/y)"""
    s = np.sign(x)
    x = abs(x)
    x = x + (int(y) >> 1)
    z = floor(x, y, b)
    return s * z


# DCT Algorithms
def d1(v, s, fe, fo):
    """Ligtenberg-Vetterli-DCT"""
    c = [0] * 8
    c[0] = flt2fix(1 / np.sqrt(2), 13)
    c[4] = flt2fix(1 / np.sqrt(2), 13)

    for ind in [1, 2, 3, 5, 6, 7]:
        c[ind] = flt2fix(np.cos(ind * np.pi / 16), 13)

    u1 = np.array([c[0]] * 4)
    u2 = np.array([c[1], c[3], c[5], c[7]])
    u3 = np.array([c[2], c[6], -c[6], -c[2]])
    u4 = np.array([c[3], -c[7], -c[1], -c[5]])
    u5 = np.array([c[4], -c[4], -c[4], c[4]])
    u6 = np.array([c[5], -c[1], c[7], c[3]])
    u7 = np.array([c[6], -c[2], c[2], -c[6]])
    u8 = np.array([c[7], -c[5], c[3], -c[1]])
    M = np.array([list(u1) + list(u1), list(u2) + list(-u2[::-1]), list(u3) + list(u3[::-1]),
                  list(u4) + list(-u4[::-1]), list(u5) + list(u5), list(u6) + list(-u6[::-1]),
                  list(u7) + list(u7[::-1]), list(u8) + list(-u8[::-1])])  # 1-D DCT

    V = M @ v

    for i in range(8):
        if i % 2 == 0:
            V[i] = fe(V[i], s[i])
        else:
            V[i] = fo(V[i], s[i])

    return V


def d2(v, s, fe, fo):
    """Loeler-Ligtenberg-Moschytz-DCT"""
    c = [0] * 8
    c[0] = 1 << 13
    c[4] = 1 << 13
    for ind in [1, 2, 3, 5, 6, 7]:
        c[ind] = flt2fix(np.sqrt(2) * np.cos(ind * np.pi / 16), 13)

    u1 = np.array([c[0]] * 4)
    u2 = np.array([c[1], c[3], c[5], c[7]])
    u3 = np.array([c[2], c[6], -c[6], -c[2]])
    u4 = np.array([c[3], -c[7], -c[1], -c[5]])
    u5 = np.array([c[4], -c[4], -c[4], c[4]])
    u6 = np.array([c[5], -c[1], c[7], c[3]])
    u7 = np.array([c[6], -c[2], c[2], -c[6]])
    u8 = np.array([c[7], -c[5], c[3], -c[1]])
    M = np.array([list(u1) + list(u1), list(u2) + list(-u2[::-1]), list(u3) + list(u3[::-1]),
                  list(u4) + list(-u4[::-1]), list(u5) + list(u5), list(u6) + list(-u6[::-1]),
                  list(u7) + list(u7[::-1]), list(u8) + list(-u8[::-1])])  # 1-D DCT

    V = M @ v

    for i in range(8):
        if i % 2 == 0:
            V[i] = fe(V[i], s[i])
        else:
            V[i] = fo(V[i], s[i])

    return V


def d3(v, s, fe, fo):
    """Arai-Agui-Nakajima-DCT"""
    c1 = flt2fix(np.cos(4 * np.pi / 16), 13)
    c2 = flt2fix(np.cos(6 * np.pi / 16), 13)
    c3 = flt2fix(np.cos(2 * np.pi / 16) - np.cos(6 * np.pi / 16), 13)
    c4 = flt2fix(np.cos(2 * np.pi / 16) + np.cos(6 * np.pi / 16), 13)

    m = [0] * 8
    for i in range(len(m)):
        if i <= 3:
            m[i] = v[i]
            m[i] += v[7 - i]
        else:
            m[i] = v[7 - i]
            m[i] -= v[i]

    n0 = m[0] + m[3]
    n1 = m[1] + m[2]
    n2 = m[1] - m[2]
    n3 = m[0] - m[3]
    n4 = m[4] + m[5]
    n5 = m[5] + m[6]
    n6 = m[6] + m[7]
    n7 = m[7]

    o0 = int(n0) << 13
    o1 = int(n1) << 13
    o2 = int(n3) << 13
    o3 = int(n7) << 13
    o4 = (n2 + n3) * c1
    o5 = (n4 - n6) * c2
    o6 = n4 * c3
    o7 = n6 * c4
    o8 = n5 * c1

    V = np.zeros(8)
    V[0] = fe(o0 + o1, s[0])
    V[2] = fe(o2 + o4, s[2])
    V[4] = fe(o0 - o1, s[4])
    V[6] = fe(o2 - o4, s[6])

    p1 = fo(o3 + o8, s[1])
    p2 = fo(o3 - o8, s[3])
    p3 = fo(o5 + o6, s[5])
    p4 = fo(o5 + o7, s[7])

    V[1] = p1 + p4
    V[3] = p2 - p3
    V[5] = p2 + p3
    V[7] = p1 - p4

    return V


def block_jpeg(I, d, s, sh, sv, Q, fq, f2, fe, fo, fs):
    """
    Compute 2D-DCT, quantization + rounding for an 8x8 pixels block
    :param I: 8x8 pixels block
    :param d: 1D-DCT function (d1, d2 or d3)
    :param s: scalar for 2D-descaling
    :param sh: 1x8 vector DCT divisors for row de-scaling
    :param sv: 1x8 vector DCT divisors for column de-scaling
    :param Q: 8x8 quantification matrix
    :param fq: rounding operator after quantization
    :param f2: rounding operator after quantization with power of 2
    :param fe: rounding operator for even DCT coefficients in 1D-DCT de-scaling
    :param fo: rounding operator for odd DCT coefficients in 1D-DCT de-scaling
    :param fs: rounding operator for 2D-DCT de-scaling
    :return: 8x8 2D-DCT output
    """
    I = I - 128  # normalize into[-128, 127]
    I = np.array(I, int)
    # 2-D block DCT from two, 1-D DCTs
    for i in range(I.shape[0]):
        I[i] = d(I[i], sh, fe, fo)
    for j in range(I.shape[1]):
        I[:, j] = d(I[:, j], sv, fe, fo)

    # Scale and quantize DCT coefficients
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            I[i, j] = fs(I[i, j], s)
            if (Q[i, j] & (Q[i, j] - 1) == 0) and (Q[i, j] != 0):  # if Q[i,j] is a power of 2
                I[i, j] = f2(I[i, j], Q[i, j])

            else:
                I[i, j] = fq(I[i, j], Q[i, j], 0)

    # np.clip(I, 0, 255)

    return I
