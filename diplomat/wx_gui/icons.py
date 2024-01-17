"""
Package stores all toolbar icons, and contains methods to convert them to bitmaps for rendering to the screen.

Icon Storage Format:
All stored as raw uint8 grayscale image data, meant to represent the transparency channel. Icons are loaded in as RGBA
wx.Bitmap objects where all pixels are of the selected foreground color, and the icon data simply controls how
transparent each location is. This means icons can match the foreground and highlight colors of your system.
Data is then compressed with zlib and encoded in base64 so icons can be directly stored as short multi-line byte
strings in the source code.
"""
import base64
import zlib
from typing import Tuple
import wx
import numpy as np


def _dump_pic(file):
    """
    Not used, but super useful for converting images to icon format used below. Just left here if I ever wanted to
    add more icons...
    """
    from PIL import Image
    import base64
    import zlib
    import numpy as np
    im = Image.open(file)
    im2 = Image.new("RGB", im.size, "white")
    im2.paste(im, (0, 0), im)
    arr = np.array(im2.convert("L"))
    arr = 255 - arr
    dat = base64.encodebytes(zlib.compress(arr.astype(np.uint8).tobytes())).decode()
    print(im.size, "\n", dat)
    return im.size, dat


# The undo toolbar icon...
BACK_ICON_SIZE = (64, 64)
BACK_ICON = b"""
eJzt1c9LFGEYB/BXjEzXLBUrMTVURIQQWSkyQq1NROggFQaKgSUUhBGCXgMDoVtBnqIOHqK/oJNH
ozp4KUyQVdGkrNyc2Xfmndmd2bd5d8d13vmxs7vPTfwe93k/D++8vxahwxykdMP4jR8g3hffhfBr
hEL8ZYlC/CWZQnx7NAHxbSLjefvzKZ6vb47oFOCbdkyen2/4Ezc5xaFQqDMYDLbVl2XN636lOSUC
yz+MFarjn98/PL/fU1uQmdds7XNH9KgYUxdn+k968jMbMW9uJi5qq9MtrrxqVfXlyShqeOq0g1eu
ZMlZZOVVtc2/9J88Nwll9gTnSz6TnBpQ8vsq16B0MccGVJk9am1Q9lXJsYE8X2xtUL6Uc4NPpV57
QEiCypIk6VSTBdGrMVkotDY4tZbeBWHUPGlFFc1dw1NzS0Q1bzYXaYZbxOr0GRQG7Cfk3NDr9Rh2
zKCPG3R27w4It+yepebhx5jEN9gu4kbUb6caCHfcPFukiU1Jt3hpkq83/NXYz+KQh0eooGdBtiwF
tl3J1BMmjnh6I1e+7H8FHrcVW3eNBtHRTB6h25F0h7C9xt5gPJbZo/I5ee8Uddhrxl8IfuDjERox
V0F95ihdwNojX4+C5nP7zVnqJI/9vbHZyb3SXB7FLv/5G2lMbrbY61I6ko1HLWwXyERWY10zyHbh
ff4evTVmsAzwgYhxWQAejWGaOAbwhWuUNEImcDcqXoT4AFGvQzx6R2+CfC+9B/IB/QnIo/BTmH/z
AuaPF/uPOcyBzn+T5jn6
"""

# The redo toolbar icon...
FORWARD_ICON_SIZE = (64, 64)
FORWARD_ICON = b"""
eJzt1U1oE0EUB/Bp42dtlQaJVQpBayu1B6HiRainXBSvioqKeO5B0INXoRcVEaTY1kIFLVhQMAhF
xRasKErVoIJgJYJYLyZB2E13Zvbzmc3X7ib7lb6j+R+H/e1m3ps3IaSZ/yaxvTi/608fzhu5HpRn
WnYnyoOWiaM8qMs7UB6UX10oD8qPrSgP8vcoyoP8rRPlgX3uCK9aoj37CzmUSJypeODvNwWxtb2H
h0cX0hkKOlvJ/xXM8IoH+majj+0+MZHWmSCBd+jL9e62dWgyq4iGDy2/4MU6Fz1wM5tXAm3pBbNr
avW+p1wOh81IjyMO3fecaeF1IfpJe5uGG9Qgjdl416uVxjTQqRaLx3+HrJr19Wkb7800yunDVovH
cl5bp4LIDFAlSeKgUdvXk7baR966dE0VNWFx/MKxg/HO0qOR6JjFHb2/WntQDVH5cv1o3TVxo/qr
5uxnr587dZ4mj7vO560KX3Cc/Tn75mX+8ZTHZJDbZe6cvSFmaYVP93vgQiaLz7BF5+zPVmdN5zO+
/y93i/yTc2+xauelpUE/Tch9c39ftzgXL5WLb7DLdfNYkxm3qzNVrlvuQIAm5JHL3d+ulmr6YVsg
J0n9Z6x27YhYLMrrDcGcPFneXrc2ohZb0haCk2vd9WvzZt/S7WE4catvxty8z5kJSFvh8NDTq+Zk
DwVtfvWcJARgAwh/Li/fQ3Bykcu7Mf6KkcJwMgrnUf6Bvhnlny2hOHl3B+dTZ3G+I8zcNtOMR/4B
kF86kw==
"""

# The save toolbar icon...
SAVE_ICON_SIZE = (64, 64)
SAVE_ICON = b"""
eJxjYBgFQMAqSAjw49GttO37f4Lg4yotHNr5HxHWDQLfArDrLyBO+///v8Kw6p9PrP7/f6Kx6V9G
tP7/f5Mo0///Xy5e/X/fYQOfkA3Iw6O/kwd7CGucQTKhFKf+k9h1gwxA9kMNLv1T0SRsSsuBIJ4N
yET2wv9mHPonooob/oUI9wDZ71BCsYso/fFQ4b0Y+v9PYCRCfwJu/f9nMVGm//8ECvX/s6ZM//85
FOo/RlB/DFR4F1b99wnql4No+peFVf8DTP2zUPUzCDq4AIEakMX0jRj9t9kZcABHDO3Y9P/fYa+E
DajFPidOPylgVD8W/fvr0/CB7Gmv8erHUjijAfFzePTvJKidgcEIj/4SIvQzvMatP4EY/ddG9Y/q
H9VPC/3lRGhnfItb/35G3PpgwPI/bv3/qwkaIH8Vn/7/Z3vK8YHqJR//49VPAkDoX0qh/may9J+A
6zf8TY7+VESgpmK2LgiBf5ORo0ouFW+wY4IcA8IpZUQAAA9i5NM=
"""

# The run toolbar icon...
RUN_ICON_SIZE = (64, 64)
RUN_ICON = b"""
eJzd1ylMA0EYBeDlCEVwGC7F6QgQJAkEXcpVSEBW1lbWVtYisSuxKzmCJaTYktYQSKqgCFoSIO1j
RSFpd45/+hQ8/5nJvH/+8bx/kR7SJ/1Jyh/iLRejPFA64jxwscR5fJ2OUR54yfR34w/wm+IW54Fg
gfP4PBmhPFBJ91EeKGxwPjyGWc6jnh+S+qTKAw/HwmJqPHCzxnk0/SnKAzVBs00eKFubbfbA5TLn
w2aPm/y+zQPV7ADlgfsE54HzRc6HzR6lPPCcUTVb7oG7Tc6HzZ7r9HtOHu/5Ycq/ZmOEb0QfWxd/
vRo9P7l/SqlGktTrRuKujAczSi30hidB4NUXV+p1xRF6bXFF3jA4WtkxaOPgsvqGP2HVBn+1ItBa
/5gSaY2v5waF3NuO6ubZtFSr/O26XEd9Jd3rwjv8h/P+1uaDeUfd5otxZ+15iR9d7W5/bnnLK2/z
ti3D7O1bjsnX5JdVkbhoy9OH/T/+2XwDmpRSew==
"""

# The jump to next bad frame toolbar icon...
JUMP_FORWARD_ICON_SIZE = (64, 64)
JUMP_FORWARD_ICON = b"""
eJxjYBgFo2AUoAHBKdfeEQDPdzjg1M514z8R4F8ALv1FxGj///8xEw79y4nT/18Gh/4tEOl7G1Zh
B8e/QxQo49P/JQJn+DDIHSKsPxu3dgYG0VeE9P/lwaefYQ4h/Q/xamcoI6T/Pn79JXTVz+xJQD9X
EF79Vc/48euXeXtEH7d+wdv/+gnof/X/+RoRXPpX/P9/Cy06MfX//3+7iQWrft1nwOTQTlj//z8X
w7DpPwqSu8FJWP///592q2Hoj/0Ekvldh11/7JnbQHDvDyxPPl/Ch6qf7RpE4hobNv3stzBy9bWK
rcj6e39DhH+XY9Ov9BGzWPj1AUm/1AOY8BVQ6LI5o+pXxqL//2ck/Vvgoj/zgNwFD7lR9Iu+wtD9
eMEOhH5bJPnLzAyRb/62oobf5r+ouj/tUUUKP8azSFLfU+XvwSISrp8luRUIJn2FBtLZYJT4y/2K
bPaFcyAltbjj/2Y1M0r64b2J4rZv4Fi+yopD/7MVIogwA+ufg+Y5MPhVjFX/50NGMB5Mv8oTLNoh
EYmRfy9HIxIHTP8BrNr//8jE0M/XhJw6ofp93mPX//8CEzHlF/sdHNr//0wnrP8Rg94nJPDlGzLY
Rrj8/svLsOIHDvtvKTHMI6T/fw4Do0MOVhArRET99f9LFA5ZIJA/8p+gfmBCW4uj/t5LoP6mtP1Q
SJz2R7jaL5zXidH+1w9n+AhMvU+o/fbyoB1O7aNgFIwMAAAzLG4P
"""

# The jump to previous bad frame toolbar icon...
JUMP_BACK_ICON_SIZE = (64, 64)
JUMP_BACK_ICON = b"""
eJxjYBgFo2BEA8cdz98RANemCODUHvjvPxHgOhcu/feJ0f7/fwUO7WLEaf+/Dod+ZYj09+OrsIMN
9yAKtuLVf1AWZ/gwRH4hqP+lCG7tDAw5BPXPwaedgf0bIf1lePUzPCCkvwS//vt01W8hRpF+lmuL
CeoPRKRmDP2NP++jOwBNv+6e1zI49Qvd+f9/Lj79ArOf/X+FW/8qIPeuIE79TEW3gBzc+vWeg/hT
cen3PfXzP179J8DZ4Q4/Vv3yaz5AMhNO/YmfwPL/+rDo557+DJaZ/9y7DQRnYtH1c16HKrjFg6E/
/RJGQXWLDU3/xL9Qmb9tIK65EEL/u5/ouv///6iEql/mPlzqOgcDg/j9uQj9nzG1//+ojKp/B0Lq
dy0D0xFYRIL075j0CEP/S1EU/Q5vkOSusU3+8f/fFKTwk1v7AVX7340o4cd0Hlny1/oXQPI2L3L8
eRyDBsLXSa1AkMSMor/kO4rpX8FWdKOkH+bqm2A5bPHPexNLCEEiEq4flPqf4tC/EGs99LeFAS3/
aO38iE2/2mNs2v//v8GBkX8jTr/F1H8Qu/b/v6sxyw+2Jj50/ZHY0gcY3GMnXH6VMez7hAS+fkMC
X6QIl99zGNRu47D+xxIG9u+E9L8SZRCOzcEK7BkYcv8T0v//sDwOaSCI/kpY//9ve3DU32uvQhTg
0k9p+4HhHnH6y3Hp9/9LWDMwO3PiDB/7gy8Jtd/uT8XdfhsFo2DkAgCaim0b
"""

# The turtle icon, used for speed adjusting control in the toolbar...
TURTLE_ICON_SIZE = (64, 64)
TURTLE_ICON = b"""
eJzt1F8oQ1EcB/CzNKP8f/AinoRMpBBKEvlT9kLxpjyJkSQkpKREQpEUZZ78yYgXscLD/CtS/ibF
k4UmMWVl/OzeM3F377k79+xN+77cs53f55x7t9+5CPniC10iDLZ5FTuPvASAXGau3nNyaGL2PRyH
TlYeY+e9ntVP8RxKGXncB/ZRjH4A80tGrrrFvpuuXJ1e0zlkWJgZ7qrN1HBfZGEOidL14a2D+vif
D3G95nf4jX2vX4v68PiQsN8sN7lf7mxOdbUZxNm/wlcdwW/g6QNt/qmE/l2G9LzNroJPOQ1feSQf
fCcLXTGSOEIl8jvjvMkcvlaaG3hO8HKBY3+ir6bx0OCs9FdLcNU1lb+rWLECPBhFjZDl2QqyHi70
7Qo9bAnfpNNKPQibyajY1wm8wcv9O5Ryt+fPUcg33H7/NJoDAHBfufoE8Ljk/v9HP9Jt2+KsDdCI
us9vh46fB4goHz0dt6VIc42FijtKpDkqxvP2V3n/GUvw9fz0QXL0pvwCIwSvtYFju4xrCN2NnL/3
IywQmhrmGgU2nkjAi0V8zSB4QdJGz/4209fFRDaqwuM6z5pPSEHb2JLJZFoeby/i76wc+0lKL0oh
9kesPgZ7C6tH+Hy8MPs13juYfRfuQFIDeIyW91bm/dEu583sXsf5AXbPveEdSV74oLmXFi+4L/8y
39uVuL0=
"""

# The help toolbar icon...
HELP_ICON_SIZE = (64, 64)
HELP_ICON = b"""
eJzN10tIVFEYB/DjaChTWjTlohaJ5ZSQMYtaRGAboxYt2tliQMKEQAiRgqBNRdhGsUUgFFKECEVC
D1AIF+EmKBe6MNJ8kTglNCrNfT+dud93RaeBe+e/sW83h/ub87z/MyPE/10Vu1BZdbVvctVx3fXZ
Z01lRevEa1O2XCpT+5MsKUrvfWGa7tbSvhwqgp9a1ty8MlJHQ/PTkp3Ps5NIHwnJa9YK8OwIxsOt
YvmM4RPFkn7rlsKf1Puh/D3/eUN9FM9+rnuo8xfqB0PwmN+7OuFPOL5CO6l0hfA3Zeafo5tt9Sq1
/S0N9sM8+OnKLY23aU7GmWD/zXvSWdu2W7tpUMqdYD/qcfXs9tYhGtWrYH9Rt11LOZfX2kkDGAv2
4vzbqfcn8xuTGc9/D+ELVrvk+UnU99EJ+Ajy0lXalX7QX6H3WWrFeNmc43m7BvMdfH7nMd6g8yvR
BvHoLAdpuhzhJUM8eu0a1P1dn49A/DIHuZOuRvhx2U+zRoRXzXH3+nWER0Z58movwsVjvoe0TyGS
799KchZbi/sQnuBz50h1CK/4wefOuIBw0csXkd4B8WO8c+ogxMVTWnt7qTL42QIV48UzE1j3SUpc
7TnGxUv6EWEcBn2Ktv4DyCPUvdwM+hgfvVrQx6l/B4q8bO2nxP8FciG+5g6/il5YQpxIyRlzHDt7
XkUvtTTgeserdUZZ6NkD8we54FV+HgB5PSWn9gT0tzi7lkHfz7eWAvpuDr8F0DdR+BkDoBfvcuuv
Z9DXV0RuLNrrb9Dw8qr4P507UBvztTl9
"""

DUMP_FRAMES_SIZE = (64, 64)
DUMP_FRAMES_ICON = b"""
eJxjYEABvKKkASFU7QwGnqQBBwz9aeXEgwIs+nf+Jx48HNVPJf1qLhDg7hkQQgyIQdOvSWK6CcDQ
P/8t8eAhFv0rSPD3x+Gjn5EJDLQ8l/4iCvxB029LYsS1UF3/7X/Eg33Y9JMQcKP6R/WTp99UCQQk
uYSA5B0y9IuxK0GBAMM1cvSbwVilg0G/N6z+JlM/DJCr3x3WfiFTvxuswhzVT7r+WRso0w+sMyjS
r66vr0uJfmSAX/+nuzAgTJb+DzpwhWTp//9QisEW0laeAhPaCuFHMTAS4//L/LznsYl/1IPZP3U5
XlDBLPkQU/svV7j7CQFdBq336Nr/xcP8r6hBEIgwOPxE01/LQBqI+oeifS4k7EgA9cja97Iykaid
kXEeQvt5HlJtBxrAAm/XP5IkXTsDAxMPNBl81CWsGKsLJB6iRDzpAJQMEBFPBnD48b+GAu0MDJGz
SY54VMBIasSjaafMdmwAAI2/hN8=
"""

SWAP_IDENTITIES_SIZE = (64, 64)
SWAP_IDENTITIES_ICON = b"""
eJztl69LQ1EUx5+8LY8tbDCb+Bc417QJJtEiNsGwF2yiYBEsBovFoNgsUzAZLGrVIjZRmDDjRINF
0DDYcb+Ud+/5nuu5e8Hiifecz+dd7q93bxD8xx9ErlwYDMyW5tcPLhtE4zyXystcemQq2j65fafv
yLKS8OhxGJDh3Nr+Rb1JZrwBnAgKKi3icYNwD8ExxPWCLYyrBUsCrhVMGul8LZaqFRUCq6ZwH8uh
HhQbBv4xFHgJjHQ77tgHnAIbp1PeQ4eA4bQDhkgUcJyWAS8JAE7TiMeCYnxun/vTOAp5JLCaeuug
mcY8F7CGrqAu4KwedKgjOBd5U1CvMbwr2JN5ONxxvCNYdfCCIL4gKrMuHgrMTRk6eSCAh4Je4Im3
BfGBpydfPCmfsP8Jxy/h/P2+fqIZbzwuiForShzun6hFuzpc3L9nSlw6Px60uHB+fdq/HxEXzk/0
i/Q5vyf0OBQseuBIsOmDA8GhF24NYjuu/HB2f3gxs/73l4yRDavur/P701ggCCAesfvbQoAFSpw2
7JqeQIvzCewK1Dhd87qw6nF/fwWVrvdDyn4/ZORaV/y8X0qD8f3IlR19/Q99fAGnbhGP
"""

SAVE_TRACKS_SIZE = (64, 64)
SAVE_TRACKS_ICON = b"""
eJxjYBhigF2WmwLd+itff/5y2I9o9XzOUT6SCG7gkzfvP37++r2DON1MiXuPnzx9ppUHyle//+LV
m3cfgAZEEaW/4sChI8dOnj67kl/XP7exf9rKx0+fvwQa8OnLNWK02+/eux9owIlTZxdfvnr91p17
lx48Ahrw+i3QAFUi9HfvABpw8DDQgKOnLl29fvPG7XsPHj15BjTg/UcbwtoZ12/bsWsP0ICjx0/N
6cgL9/M4e/suyIAXr96+VyOsn3nD5q1AA/YdABpgABaZcPP23fsPQQZcYCTC/TM2bt66fSfQgEOH
RcACmpdu3AIa8Pjpi2gitDNkrtsAMmD33gOToCJBl28Ag/Hh43ZitIvNnL9uw6YtQAO2w31ruODq
rTs7A4jRzti4fPactUADtq01QxLmVRYmRjcDg8OiJYs1PWomtkULEKcBBYgbNkxbGEiGRjBQLJs2
c/bsFiJdigG0eyZNmTZj1tweclzOwMDT3Ns/cTLQgDnFZOl37OjqARowdfqsObLk6I9vBRrQNwFo
wEwncvSnNre2dwINmDRluhc5+kMaQAZ09wINMCOsGhNo1zU0tYAM6O/hIawaEzAm1gINaOvo7nUn
RzswAlNq6xuBBoQQk8uxARaz5NqaMrKTLwio1dYXolh/5jYENBCnn7mwpl4OWeD9fwiYRqQD3Ktq
fCjRL52fXZARqkG2fouyympgNMSykadfv6ikrKIKaACsqiNNP3t6fmEx0ICaukYVcvSrZOUADSgt
BxrgT45+o4ysnLwCoAGV1Qnk6NdNTQcZUFRSXhlFjn7x5NT0zGyQAWW25Ohn9E1KARqQm19UwEeO
fgaB6KSUtAygAboMZOlnEPBOBBqQoALjk6qfgTE4PFKDGc4lWb9EZEwoiOYRBIMPUP1zIVw2vHrZ
xKT4LcIjjUFs2x//McE1fFUTq563X2BwgF8YP5ibgKn9jTI+y63dvXz8AoJC7aEC3ejaf9rj0c6g
7ezq7ukNNCCQCyLAtAFV+79YfNo57BycXIAG+PoHweKe5wKK/mas+lg4mMC0qDXIADcPoAFwZ0o9
QdK+lgmLbgElLR1dZZCDxS2tbe1BBnj5IqpP469w7WexdQBElVTUNXX09IHRImBmATTA0RloAFL1
F/IPqv2pDBbtvDLyiipqmtp6euwcMsZmFlY2QANc3ZEL8HqI9m+mWLQziknJyCkqq2lo62poaOka
mwINsHNwtkT2KOMSkPa/WNt+LCLikjJyCsqqGlpaqupamsam5pY2dmYcKIo4jgH1l2PTzsAqKCIm
KS2noATUrCDBycCjbGCiK40ezCJ3/i/Aqp2BmU9QWExCWhZogBKeOldrCzsOGW4+AWFRCSlZeSUR
3NrxABagAUKi4lIyssyEFWMDrNy8IAMkOAgrxQ6YOXj5BXlZyNUOAuS2VgYSAAD00GPz
"""

SETTINGS_SIZE = (64, 64)
SETTINGS_ICON = b"""
eJzll11oFFcUgG/WNSEBKUJqjGti0gi1IoXaSOJGLNo+WSTZTX8otS+aGHzoj8aArdDKsomhD6WF
UgMVXWkL27R9K8UaLLTrg5jfTZHGNjQ2G92wuzFKfpowd+7pmTtzZ+/MzizbvHpeZu495zv37vm5
c5eQx0CKvP8TqH7nG480PDAfaS4tGPa0/MoAXpFmoiqFf6NFBfK/AAWgw9mJCgVQ/ix0/TDTzKHF
nDinTdCvC+Wf5jhN+ozxzmXu8FShPImr3EGitQQHpW9lKHf4QsH8ZzpAYfnOyOQKcG8AdQXzYYOw
yrMF8z9TJ76zUNzPnHD1QaVpUeTJg9clHZcHdaJKLHDz/RzqyPWgXuR+FxzjmW7VDGq+B6Y8b8Mr
5hkkQ43V/j7FgqdS8g5g8KOTUQWjSyfWW/koTjLGbYSkL7dUFRNSXBW4nM56AMP92xZ8d07IEkfX
ZdXeYzM2NXuwQeZ/tPFqyNauZWGLhcrOyFrPl8xSMgu8d9btO98fi/X3NPFsBRelSK68aovfoVnJ
/4IWXW+HuefEcS019QsmPpVbi0/0gfCgaqvXjZuxxEf8KZwKGGN6d3MOjvKecB/CQVMGAz14qt7n
23N6CF8zfpwMGwb7nXBCYnpqZsoIqU0BvX9YKJqxpOa2YytPc4MpZ1xs4Cj+9nGgo9lyJ5VjFOIY
gzaeumsu/GscT2PeO3D1Slm1BXfQjn4zmsWgC3+C85cwcRj5w1ZdC2YB0xjRLB65dN8nPL7NhOwD
OmRXDquAIQzwJRqd+d95AqsJOe9wWJwG6Mbe4/n8whHfrxc2tsx3AHvs2gaAb7GZ+BLqAQd86xRP
XwpfbwD4ctQAMXzwTqSroe029ebOh3r20ziIOfO/4SOj55jB/I0LAdFju84OM1GsDM/8fuf9Rwkp
yXYJo1CvqxpB7j6MX49D/LoAwkb8TNmkq560TGLzNIGak78RFfZiE8uWq+KD/EiejeBxkOBlIAsm
fhrL5opkyMaE8pp8ZKaxzo/j19NSv75ZCm1Yv3OSofq50H6gB88IwjG0iwMd2yLhcQqj2BftxsrK
0MBQir0p1Dt6e0++/tIbX6ncwz3s3xrs36R5AQji6nP4+SxN6HzfRm22OOdS9PIqTw/Gmfi182O4
q8Hna+gawdc0Bk/LiyYf2zlTOvXtBfC1Li6fX6O1ONXKvdOR9a689w+OLGiF4W1PiEhNt2nfgnrj
AD7oihsHDMCitgPi8XdHY7Fo917e7606zsbz4KRc5DZcZlWU9RiVqzg3r5BlseeZNim83vZ7Yl65
kA/fKH1FMpHgNmymkm3BK1LZsL/z3R1OgE0s329dOtzxTRnHm5NF1OUdbrjnqjvOFMX4bXS8xIX/
0DCAiYE7AFIoGGO3L12Mi7tDrzO+Ia2fjpPa4bP1TNJ0QJdC1ZrBoTnNAU26/YCd2tWJ/lVuuPtB
bHhanJbP4FVWvVvrGr+a20ydNdXFcWO/L5oGuxdhoNwVxw69+HBXdnREx+9LfxwOnsuXfpQK6d2n
J+yn/IS7FK1ovBJZK08mefg+XTN/lWfg3TXzZ6eXlv659dya+cdA/gPgZyV7
"""

SAVE_CONT_ICON_SIZE = (64, 64)
SAVE_CONT_ICON = b"""
eJxjYBgFEKBeue7oGUwwj40o3cKL//7HDjaxE6Fd+Q5Q5QljDNABFN7GQVA7/3WQTRuRhXhaV83R
Y0gEie/iIqS/6z+G/jqgwBWI/v+HePFr5/mGqX86UOA9VD8hA4L+E9D//xg/Pv01BPX/PyuMR/9E
wvrxGjCZCP3/r0kS0r8FWWwaUOAtQzJSQrouRUD/XSYksdB///+vYpiAnBJvSuPX/78OWVDHxY7V
/AdKUr6viF///43+6koI4Njx/T8qeKCEXz9BMGlY6//7/C528OIfEfr/TRXBKgsCUksJ62/DqRsE
5hHS/x5/OSf8i4D+w/i1Wz4FKZqMW/8enHoZrafehIbfjw2x3CTqZwy5gBKF75sFSNGvvA8jETwP
I15/wAdsyWgmK5H6U/9gT4fbOYnSH4WrUvu/iYUI/Xro+R8JdBDWz3ELt/b//xwJ6q9H1TEbNSyu
sxLQz48W9OIW11H4iaj6T6LrL0NzsTQDZweyE67AVXaCuD/R64eraPrlgGJWN5EEjGEq88HcLaiJ
Qgk9xMBlL/dkRJTWw5QaQPjnApBL71J0/SoQxUZnYAL7YPoZL6OrhYLj8kgA5jyWvM8Q2Rdwt3rh
0L8fI04hXtsBkUa0azpJ0g+sHV+CpCXgfMaGfyTptwEnBeQK2WTLT6L1C0yARAIPiii/bUgoEtiK
U7/PE4jpb3G5DgzS0B2jChFX3A4TOIJXvya6fnD6YSr+Chdoxauf8R6aflD6VTqAJGCLVz9DE5p+
aQbGtC9I/PuM+PVLfEPVL6GIWhTn49fOgNp4AraOvqBwHxFsVfM/Qw9CZBBESDsDgyOO0hsE5hHW
zsBQjFP7cYKuBwP0OIABvM1pZJCFJVcAiyoCHQokYHYNQ/fXYgIxjwJYC5+j6P69UJ4E3SDAEbkJ
WmL9/3u2Wo5E3WDAYhhRUF6W6CBIjmZaAgCitspG
"""


def to_wx_bitmap(icon_bytes: bytes, icon_size: Tuple[int, int], fg_color: wx.Colour, bitmap_size: Tuple[int, int] = None):
    """
    Converts an icon in this module into a wx.Bitmap for rendering in wxWidget's UIs.

    :param icon_bytes: The bytes of the icon provided by this module.
    :param icon_size: The size of the icon, also provided by this module.
    :param fg_color: A wx.Colour. This is the foreground color that will be used when rendering the icon.
    :param bitmap_size: A optional tuple of integers which defines the size of the returned bitmap. Defaults to the
                        size of the original icon.
    """
    if(bitmap_size is None):
        bitmap_size = icon_size
    out_w, out_h = bitmap_size
    w, h = icon_size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    alpha_mask = np.frombuffer(zlib.decompress(base64.b64decode(icon_bytes)), dtype=np.uint8).reshape((h, w))

    img[:, :] = [fg_color.Red(), fg_color.Green(), fg_color.Blue()]

    img = wx.Image(w, h, img, alpha_mask)
    img.Rescale(out_w, out_h, wx.IMAGE_QUALITY_HIGH)

    return wx.Bitmap(img)


def _main():
    # Tests icon loading code above by displaying the help icon in a wx Frame...,
    app = wx.App()
    frame = wx.Frame(None, title="Image Test!")
    bitmap = to_wx_bitmap(SAVE_CONT_ICON, SAVE_CONT_ICON_SIZE, frame.GetForegroundColour())
    icon = wx.StaticBitmap(frame, wx.ID_ANY, bitmap)
    frame.Show(1)
    app.MainLoop()


if(__name__ == "__main__"):
    _main()