import numpy as np

from utils import convert_pt_to_polytope

s = np.array([0.0005147557017792792, 3.767372632249206])

t = np.array([-2.63396063704866, -3.2032454407254596])

A_s, b_s = convert_pt_to_polytope(s, eps=1e-6)
A_t, b_t = convert_pt_to_polytope(t, eps=1e-6)

A0 = np.array([[-1.0, -0.0], [-0.19611613513818435, -0.98058067569092], [-0.98058067569092, 0.19611613513818443], [-0.44721359549995954, 0.8944271909999152], [0.9805806756909201, 0.1961161351381842], [0.316227766016838, 0.9486832980505139], [0.7071067811865475, 0.7071067811865475], [0.16439898730535765, -0.9863939238321436], [-0.0, -1.0], [0.7999999999999998, 0.6000000000000003], [0.8574929257125444, 0.514495755427526]])

b0 = np.array([5.000000000000009, 5.685386947945345, 4.437375178884172, 0.9260483543181116, -2.00078077262187, -2.651202482767421, -3.1426968052735362, 4.6579713069851305, 5.000000000000009, -3.0606060606060526, -2.9102790206001403])

A1 = np.array([[-0.8479983040050884, 0.5299989400031795], [-0.16439898730535765, 0.9863939238321436], [0.9397934234884371, 0.34174306308670416], [0.4190581774617472, 0.9079593845004515], [0.7682212795973754, 0.6401843996644803], [-0.8944271909999162, -0.44721359549995743], [-0.11043152607484627, -0.9938837346736189], [-0.4472135954999579, -0.8944271909999159], [0.14142135623730917, -0.9899494936611667], [0.393919298579168, -0.9191450300180578], [-1.0, 0.0], [-0.9863939238321436, 0.16439898730535765], [-0.5547001962252275, 0.8320502943378448], [-0.3713906763541037, 0.9284766908852593], [-0.707106781186549, -0.7071067811865459], [-0.7893522173763259, -0.6139406135149209], [0.9284766908852597, -0.3713906763541029], [1.0, -0.0], [0.7592566023652966, -0.6507913734559687], [0.7808688094430303, -0.6246950475544245]])

b1 = np.array([4.898475051544551, 3.4291303917733704, 0.7205945900944476, 2.1482022733518917, 1.299768326591529, 4.585068681135942, 2.5990450076201337, 3.41056832931787, 2.0427529234278134, 1.5120134692937832, 5.000000000000009, 4.99009047325858, 4.35635760166784, 3.8733421549051807, 4.071220861377103, 4.234684174027886, 0.5158203838251523, 0.4545454545454627, 0.7066674004698736, 0.6704429171985703])

A2 = np.array([[-0.9938837346736189, -0.11043152607484676], [-1.0, 0.0], [-0.9486832980505141, 0.31622776601683733], [-0.24253562503633322, 0.9701425001453319], [-0.7071067811865475, 0.7071067811865475], [0.8944271909999166, 0.4472135954999564], [1.0, 0.0], [0.6401843996644798, 0.768221279597376], [-0.14142135623730923, -0.9899494936611665], [0.7071067811865475, -0.7071067811865475], [-0.0, -1.0], [0.0, 1.0], [0.44721359549995876, 0.8944271909999155], [-0.6000000000000003, -0.7999999999999998], [-0.37139067635410355, -0.9284766908852593]])

b2 = np.array([-0.49080678255486765, -0.6565656565656509, -1.0221503548018942, -0.551217329628024, -1.0713739108887024, 2.507106520227046, 3.2828282828282878, 1.4549645446920048, 2.6141523425684547, 3.9283710065919357, 2.9797979797979854, -0.050505050505044544, 0.9260483543181023, 1.444444444444449, 2.0539029828673976])

A3 = np.array([[0.9486832980505135, 0.3162277660168388], [-1.0, 0.0], [-0.37139067635410483, 0.9284766908852589], [0.16439898730535918, -0.9863939238321435], [0.8944271909999159, -0.4472135954999579]])

b3 = np.array([1.7568209223157756, 0.050505050505056374, 3.7232853159742256, -3.246464850322957, -1.016394535227171])

A4 = np.array([[-1.0, 0.0], [0.554700196225228, 0.8320502943378445], [-0.7071067811865476, -0.7071067811865476], [0.9486832980505135, -0.3162277660168388], [0.5547001962252303, -0.8320502943378429]])

b4 = np.array([-0.4545454545454481, 3.4038421132002776, -2.4998724587403136, -0.06388439717511603, -2.031099203349946])

A5 = np.array([[0.9557790087219499, 0.29408584883752364], [0.5368754921931593, 0.8436614877321076], [1.0, 0.0], [-1.0, 0.0], [-0.9863939238321439, 0.1643989873053561], [-0.16439898730535635, 0.9863939238321439], [-0.8320502943378445, -0.554700196225228], [-0.7999999999999998, -0.6000000000000003], [-0.14142135623730986, -0.9899494936611665], [-0.5547001962252294, -0.8320502943378435], [0.7071067811865476, -0.7071067811865476], [-0.768221279597376, 0.6401843996644798], [-0.5547001962252288, 0.8320502943378438], [0.3162277660168388, -0.9486832980505135], [0.4472135954999576, -0.894427190999916]])

b5 = np.array([4.556102733884361, 2.9361588966985273, 5.000000000000009, -1.6666666666666583, -1.7187166854650904, 0.672541311703746, -0.7984321006272179, -0.6767676767676674, 1.957043010556715, 0.43423500209551513, 4.714045207910326, -1.260969272066391, -0.6303411320741142, 3.4497574474564248, 3.8171261434087387])

A6 = np.array([[0.4472135954999579, -0.8944271909999159], [1.0, 0.0], [0.8944271909999156, -0.4472135954999583], [-0.7474093186836599, 0.6643638388299198], [-1.0, -0.0], [-0.39391929857916735, 0.9191450300180581], [0.9922778767136676, 0.12403473458920873], [0.9486832980505137, 0.31622776601683833], [0.6401843996644794, 0.7682212795973763], [0.7432941462471664, 0.6689647316224496], [-0.98058067569092, -0.19611613513818435], [-0.9701425001453319, -0.2425356250363335], [-0.09053574604251874, -0.9958932064677037], [-0.31622776601683583, -0.9486832980505144], [-0.5812381937190967, -0.8137334712067348], [-0.44721359549995876, -0.8944271909999155], [0.12403473458920873, 0.9922778767136676], [0.11043152607484676, 0.9938837346736189]])

b6 = np.array([3.2750490579542455, -0.0505050505050416, 1.0615676256817284, 3.3679555718461307, 5.000000000000009, 1.9762281309190646, -0.3194834072752256, -0.5749595745760608, -0.7824475995899106, -0.897208590672076, 5.526909262985195, 5.64691531018938, 5.102923867851053, 5.685711348585575, 6.117678766215144, 5.940261394772179, 0.23178207978792245, 0.26771279048448426])

A7 = np.array([[0.9863939238321436, -0.16439898730535765], [0.7071067811865476, 0.7071067811865476], [0.0, 1.0], [-0.8944271909999171, -0.4472135954999556], [-0.8320502943378432, -0.5547001962252299], [-0.09950371902099908, -0.995037190209989], [-0.19611613513818468, -0.98058067569092], [0.6246950475544241, -0.7808688094430304], [0.707106781186549, -0.7071067811865459], [-0.9863939238321439, 0.16439898730535624], [-0.8944271909999155, 0.44721359549995876], [-0.44721359549995954, 0.8944271909999152], [-0.5547001962252294, 0.8320502943378435]])

b7 = np.array([2.6652563093444357, 0.2856997095703288, -1.0606060606060541, 1.7391639824998368, 2.031099203349964, 3.6233424956636533, 3.5261284903633148, 3.8727937922881672, 3.785521151806774, 0.09133277072520783, -0.5646636306817602, -1.1519138065907941, -1.1065988763079002])

A8 = np.array([[-0.9615239476408233, 0.2747211278973778], [-1.0, -0.0], [-0.8944271909999159, -0.4472135954999579], [-0.3162277660168388, -0.9486832980505135], [-0.5144957554275266, -0.8574929257125441], [0.9761870601839527, -0.21693045781865639], [-0.0, -1.0], [-0.3162277660168363, 0.9486832980505143], [0.9701425001453317, 0.24253562503633397], [0.393919298579169, 0.9191450300180573], [0.09053574604251796, 0.995893206467704], [-0.0, 1.0], [0.7071067811865475, -0.7071067811865475], [0.7525766947068776, -0.6585046078685184], [0.44721359549995876, -0.8944271909999155], [0.31622776601683833, -0.9486832980505137]])

b8 = np.array([4.8908685648396935, 3.8888888888888973, 1.91985634431801, -1.0221503548018893, -0.38110796698334404, -0.5642383120030624, -1.7676767676767586, 5.8134801429358145, 1.3596694130824858, 4.323827991138344, 4.837718147120445, 5.000000000000009, -2.1427478217774083, -2.0477307791582264, -2.2360679774997814, -2.17206950395403])

A9 = np.array([[0.995037190209989, -0.09950371902099829], [0.0, 1.0], [0.9863939238321439, 0.16439898730535635], [1.0, -0.0], [-0.9615239476408232, -0.2747211278973784], [-1.0, 0.0], [-0.242535625036333, -0.970142500145332], [0.24253562503633297, -0.9701425001453319], [-0.09053574604251796, 0.995893206467704], [-0.1520571842539414, 0.9883716976506172], [-0.4472135954999584, -0.8944271909999156], [-0.4961389383568339, -0.8682431421244592], [-0.7371541402007415, -0.6757246285173463], [-0.7071067811865476, -0.7071067811865476]])

b9 = np.array([4.7992955386390985, 5.000000000000009, 5.654328805805476, 5.000000000000009, -1.1307965617998064, -0.1515151515151428, -1.5311592489667396, 0.4532231376941661, 4.74626789859265, 4.61931294589625, -2.055375615681617, -2.1361537623696933, -2.205877919540089, -2.214172749169989])

As = {
    "s": A_s,
    "t": A_t,
    0: A0,
    1: A1,
    2: A2,
    3: A3,
    4: A4,
    5: A5,
    6: A6,
    7: A7,
    8: A8,
    9: A9,
}

bs = {
    "s": b_s,
    "t": b_t,
    0: b0,
    1: b1,
    2: b2,
    3: b3,
    4: b4,
    5: b5,
    6: b6,
    7: b7,
    8: b8,
    9: b9,
}

n = A0.shape[1]
