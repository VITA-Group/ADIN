Nsample = {
    'MSMT17': {'val': 2373, 'gallery': 82161, 'query': 11659, 'train': 30248},
    'DukeMTMC': {'val': 702, 'gallery': 17661, 'query': 2228, 'train': 15820},
    'Market1501': {'val': 751, 'gallery': 19732, 'query': 3368, 'train': 12185},
    'mix': {'val': 3826, 'gallery': 119554, 'query': 17255, 'train': 58253},
    'VeRi': {'query': 1678, 'val': 594, 'gallery': 11579, 'train': 37152},
    'aic19': {'query': 1052, 'val': 344, 'gallery': 18290, 'train': 36591},
    'VeID': {'query': 13164, 'gallery': 95057, 'val': 13164, 'train': 100182}
}

Nclass = {
    'MSMT17': {'val': 1041, 'gallery': 3060, 'query': 3060, 'train': 1041},
    'DukeMTMC': {'val': 702, 'gallery': 1110, 'query': 702, 'train': 702},
    'Market1501': {'val': 751, 'gallery': 752, 'query': 750, 'train': 751},
    'mix': {'val': 2494, 'gallery': 4922, 'query': 4512, 'train': 2494},
    'VeRi': {'query': 200, 'val': 575, 'gallery': 200, 'train': 575},
    'aic19': {'query': 1, 'val': 333, 'gallery': 1, 'train': 333},
    'VeID': {'query': 13164, 'gallery': 13164, 'val': 13164, 'train': 13164}
}

NcamId = {
    'MSMT17': 15,
    'DukeMTMC': 8,
    'Market1501': 6,
    'mix': 29,
    'VeRi': 20,
    'aic19': 36,
    'VeID': 2
}

labelDist = {
    'MSMT17': {'camId': [4910, 203, 454, 1614, 4296, 1678, 3453, 795, 1396, 655, 3154, 1364, 3635, 3876, 1138],
               'ts': [14594, 8133, 9894],
               'comb': [[2168, 33, 10, 361, 1718, 271, 1429, 61, 324, 47, 354, 333, 524, 1415, 846],
                        [595, 32, 27, 327, 915, 567, 709, 247, 360, 497, 1264, 319, 1366, 718, 190],
                        [2147, 138, 417, 926, 1663, 840, 1315, 487, 712, 111, 1536, 712, 1745, 1743, 102]]},
    'DukeMTMC': {'camId': [2809, 3009, 1088, 1395, 1685, 3700, 1330, 1506],
                 'ts': []},
    'Market1501': {'camId': [2017, 1709, 2707, 920, 2338, 3245],
                   'ts': []},
    'VeRi': {'camId': [2277, 2122, 2271, 2231, 903, 2034, 749, 1715, 1790, 2040,
                       2061, 2273, 2190, 2728, 2644, 2411, 2603, 173, 2509, 22],
             'ts': []},
    'aic19': {'camId': [1002, 1191, 1323, 1314, 1599, 948, 304, 97, 344, 322, 10, 3180,
                        2278, 1659, 382, 1317, 1605, 2206, 1886, 436, 1958, 2442, 2198,
                        1137, 678, 413, 151, 321, 900, 1139, 748, 409, 143, 254, 116, 181],
              'ts': []},
    'VeID': {'camId': [],
             'ts': []},
}

mean_std = {
    'MSMT17': [([0.3128608763217926, 0.2919407784938812, 0.30335408449172974],
                [0.22551032900810242, 0.21623282134532928, 0.20708084106445312]),
               ([0.31065645813941956, 0.28985247015953064, 0.3041130006313324],
                [0.22369679808616638, 0.21412236988544464, 0.20583610236644745]),
               ([0.3104613423347473, 0.29011523723602295, 0.3034765422344208],
                [0.2252582311630249, 0.2158508151769638, 0.20692187547683716]),
               ([0.3100127577781677, 0.28943586349487305, 0.3027054965496063],
                [0.22489048540592194, 0.21564623713493347, 0.2067936211824417])],
    'DukeMTMC': [([0.44009241461753845, 0.43117815256118774, 0.44590505957603455],
                  [0.19992396235466003, 0.20297777652740479, 0.1905352920293808]),
                 ([0.43644261360168457, 0.4246115982532501, 0.4432300329208374],
                  [0.19644396007061005, 0.19887173175811768, 0.19013218581676483]),
                 ([0.42361190915107727, 0.4158657491207123, 0.43501266837120056],
                  [0.19978368282318115, 0.2018384039402008, 0.19105567038059235]),
                 ([0.42361190915107727, 0.4158657491207123, 0.43501266837120056],
                  [0.19978368282318115, 0.2018384039402008, 0.19105567038059235])],
    'Market1501': [([0.4147369861602783, 0.38889187574386597, 0.3841992914676666],
                    [0.19707627594470978, 0.19072225689888, 0.1892048716545105]),
                   ([0.4101223647594452, 0.38640642166137695, 0.3802562654018402],
                    [0.19489040970802307, 0.18943087756633759, 0.18800492584705353]),
                   ([0.4126043915748596, 0.3893193304538727, 0.38096344470977783],
                    [0.1902138590812683, 0.18543975055217743, 0.18257704377174377]),
                   ([0.4173775315284729, 0.3923223912715912, 0.3827044367790222],
                    [0.19056615233421326, 0.18477684259414673, 0.18223266303539276])],

    'MSMT17_histeq': [([0.35204121470451355, 0.3286859691143036, 0.3427477478981018],
                       [0.2555672526359558, 0.2448694407939911, 0.23757928609848022]),
                      ([0.3503515124320984, 0.32698363065719604, 0.34398964047431946],
                       [0.2547372579574585, 0.24363984167575836, 0.23775124549865723]),
                      ([0.34888848662376404, 0.32606077194213867, 0.3422895669937134],
                       [0.2548025846481323, 0.2440209984779358, 0.23713767528533936]),
                      ([0.34900009632110596, 0.32588011026382446, 0.3419249951839447],
                       [0.2552053928375244, 0.2444157600402832, 0.23749038577079773])],
    'DukeMTMC_histeq': [([0.4741177260875702, 0.46506068110466003, 0.48055019974708557],
                         [0.2344294637441635, 0.23673708736896515, 0.2254134863615036]),
                        ([0.4703892171382904, 0.4582761526107788, 0.4776722192764282],
                         [0.23232653737068176, 0.2336699217557907, 0.22608982026576996]),
                        ([0.46758371591567993, 0.4579831063747406, 0.4766121506690979],
                         [0.2313084751367569, 0.23207473754882812, 0.22419388592243195]),
                        ([0.46510395407676697, 0.45724818110466003, 0.4783724248409271],
                         [0.23608438670635223, 0.23758763074874878, 0.2279679775238037])],
    'Market1501_histeq': [([0.4547954499721527, 0.42706629633903503, 0.4217660129070282],
                           [0.23595194518566132, 0.22771354019641876, 0.2256421595811844]),
                          ([0.452191025018692, 0.4264988899230957, 0.41953131556510925],
                           [0.2344713658094406, 0.22709611058235168, 0.22493018209934235]),
                          ([0.45532694458961487, 0.42986416816711426, 0.42081600427627563],
                           [0.23072053492069244, 0.22407305240631104, 0.2206859290599823]),
                          ([0.4582526385784149, 0.4311460554599762, 0.42069900035858154],
                           [0.23126783967018127, 0.22336162626743317, 0.22011598944664001])],

    'MSMT17_ace': [([0.4588722288608551, 0.45648476481437683, 0.4674382507801056],
                    [0.2648240327835083, 0.2626720070838928, 0.25700950622558594]),
                   ([0.4594220519065857, 0.456997811794281, 0.4687672257423401],
                    [0.26447123289108276, 0.262509286403656, 0.256829172372818]),
                   ([0.4578744173049927, 0.4555986821651459, 0.46711766719818115],
                    [0.26468774676322937, 0.2627008259296417, 0.2570514678955078]),
                   ([0.45759427547454834, 0.4553059935569763, 0.4668932557106018],
                    [0.2646131217479706, 0.26255136728286743, 0.25682559609413147])],
    'DukeMTMC_ace': [([0.5029980540275574, 0.5036572813987732, 0.5063640475273132],
                      [0.25283098220825195, 0.25402215123176575, 0.2515082061290741]),
                     ([0.502792239189148, 0.5017712116241455, 0.505958080291748],
                      [0.25271642208099365, 0.2533477246761322, 0.25159162282943726]),
                     ([0.5019697546958923, 0.5012686848640442, 0.505262553691864],
                      [0.25172752141952515, 0.25228434801101685, 0.25026965141296387]),
                     ([0.5008956789970398, 0.5008492469787598, 0.5044423937797546],
                      [0.2526385188102722, 0.2535822093486786, 0.25124552845954895])],
    'Market1501_ace': [([0.4919191896915436, 0.4919387400150299, 0.4942207634449005],
                        [0.25826209783554077, 0.25677618384361267, 0.2560907006263733]),
                       ([0.49079596996307373, 0.4912499785423279, 0.49343279004096985],
                        [0.25840067863464355, 0.256965309381485, 0.2561896741390228]),
                       ([0.49143239855766296, 0.4913090467453003, 0.49295473098754883],
                        [0.2572576701641083, 0.2562216818332672, 0.25529101490974426]),
                       ([0.4912150502204895, 0.49113619327545166, 0.4929448068141937],
                        [0.25509941577911377, 0.2537045180797577, 0.2525354325771332])],

    'MSMT17_histeq2': [([0.4992844760417938, 0.499462366104126, 0.4993959069252014],
                        [0.29048892855644226, 0.29003655910491943, 0.2899274528026581]),
                       ([0.49947935342788696, 0.4996398985385895, 0.49951624870300293],
                        [0.2901484966278076, 0.28973379731178284, 0.2897079885005951]),
                       ([0.49894243478775024, 0.4992033839225769, 0.49910154938697815],
                        [0.29067566990852356, 0.29021644592285156, 0.29017236828804016]),
                       ([0.4990062713623047, 0.49921682476997375, 0.49913203716278076],
                        [0.2908245027065277, 0.29039233922958374, 0.2903355360031128])],
    'DukeMTMC_histeq2': [([0.5003214478492737, 0.5004384517669678, 0.5006325840950012],
                          [0.2891704738140106, 0.2893352210521698, 0.28964316844940186]),
                         ([0.5003942251205444, 0.50016850233078, 0.5005102157592773],
                          [0.2892981767654419, 0.28894931077957153, 0.28947991132736206]),
                         ([0.5003296732902527, 0.5002579092979431, 0.5005816221237183],
                          [0.28918683528900146, 0.28906142711639404, 0.2895433008670807]),
                         ([0.5002561211585999, 0.5002831816673279, 0.5005204081535339],
                          [0.28906819224357605, 0.28909066319465637, 0.289455384016037])],
    'Market1501_histeq2': [([0.5000327229499817, 0.5000055432319641, 0.5000789761543274],
                            [0.2887571156024933, 0.28869274258613586, 0.2888254225254059]),
                           ([0.500032901763916, 0.5000060796737671, 0.5000737309455872],
                            [0.28874969482421875, 0.28869208693504333, 0.2888084352016449]),
                           ([0.5000117421150208, 0.5000044107437134, 0.5000336170196533],
                            [0.28872057795524597, 0.2886866331100464, 0.2887561619281769]),
                           ([0.5000343918800354, 0.5000113248825073, 0.5000633597373962],
                            [0.2887536585330963, 0.28867754340171814, 0.2888003885746002])],

    'MSMT17_gray': [([0.295314222574234, 0.295314222574234, 0.295314222574234],
                     [0.21539686620235443, 0.21539686620235443, 0.21539686620235443]),
                    ([0.29340019822120667, 0.29340019822120667, 0.29340019822120667],
                     [0.21335332095623016, 0.21335332095623016, 0.21335332095623016]),
                    ([0.29350870847702026, 0.29350870847702026, 0.29350870847702026],
                     [0.2150428742170334, 0.2150428742170334, 0.2150428742170334]),
                    ([0.29287877678871155, 0.29287877678871155, 0.29287877678871155],
                     [0.21479995548725128, 0.21479995548725128, 0.21479995548725128])],
    'DukeMTMC_gray': [([0.4321865141391754, 0.4321865141391754, 0.4321865141391754],
                       [0.1988847553730011, 0.1988847553730011, 0.1988847553730011]),
                      ([0.42651793360710144, 0.42651793360710144, 0.42651793360710144],
                       [0.1952795684337616, 0.1952795684337616, 0.1952795684337616]),
                      ([0.42399024963378906, 0.42399024963378906, 0.42399024963378906],
                       [0.19353587925434113, 0.19353587925434113, 0.19353587925434113]),
                      ([0.4169471561908722, 0.4169471561908722, 0.4169471561908722],
                       [0.19839763641357422, 0.19839763641357422, 0.19839763641357422])],
    'Market1501_gray': [([0.39219391345977783, 0.39219391345977783, 0.39219391345977783],
                         [0.189619779586792, 0.189619779586792, 0.189619779586792]),
                        ([0.389157235622406, 0.389157235622406, 0.389157235622406],
                         [0.18836873769760132, 0.18836873769760132, 0.18836873769760132]),
                        ([0.3918055593967438, 0.3918055593967438, 0.3918055593967438],
                         [0.1842091977596283, 0.1842091977596283, 0.1842091977596283]),
                        ([0.3950938582420349, 0.3950938582420349, 0.3950938582420349],
                         [0.18357495963573456, 0.18357495963573456, 0.18357495963573456])],

    'MSMT17_histeqgray': [([0.33276858925819397, 0.33276858925819397, 0.33276858925819397],
                           [0.24411551654338837, 0.24411551654338837, 0.24411551654338837]),
                          ([0.3312743902206421, 0.3312743902206421, 0.3312743902206421],
                           [0.24300402402877808, 0.24300402402877808, 0.24300402402877808]),
                          ([0.3301871120929718, 0.3301871120929718, 0.3301871120929718],
                           [0.2432814985513687, 0.2432814985513687, 0.2432814985513687]),
                          ([0.3300589323043823, 0.3300589323043823, 0.3300589323043823],
                           [0.24365516006946564, 0.24365516006946564, 0.24365516006946564])],
    'DukeMTMC_histeqgray': [([0.46614670753479004, 0.46614670753479004, 0.46614670753479004],
                             [0.2328590601682663, 0.2328590601682663, 0.2328590601682663]),
                            ([0.4602949321269989, 0.4602949321269989, 0.4602949321269989],
                             [0.2303454577922821, 0.2303454577922821, 0.2303454577922821]),
                            ([0.45941492915153503, 0.45941492915153503, 0.45941492915153503],
                             [0.22903317213058472, 0.22903317213058472, 0.22903317213058472]),
                            ([0.45848435163497925, 0.45848435163497925, 0.45848435163497925],
                             [0.23425713181495667, 0.23425713181495667, 0.23425713181495667])],
    'Market1501_histeqgray': [([0.43071067333221436, 0.43071067333221436, 0.43071067333221436],
                               [0.22691960632801056, 0.22691960632801056, 0.22691960632801056]),
                              ([0.42959704995155334, 0.42959704995155334, 0.42959704995155334],
                               [0.22632770240306854, 0.22632770240306854, 0.22632770240306854]),
                              ([0.432746946811676, 0.432746946811676, 0.432746946811676],
                               [0.22312770783901215, 0.22312770783901215, 0.22312770783901215]),
                              ([0.43428143858909607, 0.43428143858909607, 0.43428143858909607],
                               [0.22248652577400208, 0.22248652577400208, 0.22248652577400208])],

    'MSMT17_acegray': [([0.45583832263946533, 0.45583832263946533, 0.45583832263946533],
                        [0.25991392135620117, 0.25991392135620117, 0.25991392135620117]),
                       ([0.4564164876937866, 0.4564164876937866, 0.4564164876937866],
                        [0.25963082909584045, 0.25963082909584045, 0.25963082909584045]),
                       ([0.4549664258956909, 0.4549664258956909, 0.4549664258956909],
                        [0.25988033413887024, 0.25988033413887024, 0.25988033413887024]),
                       ([0.45468682050704956, 0.45468682050704956, 0.45468682050704956],
                        [0.25969573855400085, 0.25969573855400085, 0.25969573855400085])],
    'DukeMTMC_acegray': [([0.5017633438110352, 0.5017633438110352, 0.5017633438110352],
                          [0.2503322660923004, 0.2503322660923004, 0.2503322660923004]),
                         ([0.5003405809402466, 0.5003405809402466, 0.5003405809402466],
                          [0.24990351498126984, 0.24990351498126984, 0.24990351498126984]),
                         ([0.499755859375, 0.499755859375, 0.499755859375],
                          [0.24901464581489563, 0.24901464581489563, 0.24901464581489563]),
                         ([0.4991673231124878, 0.4991673231124878, 0.4991673231124878],
                          [0.25030645728111267, 0.25030645728111267, 0.25030645728111267])],
    'Market1501_acegray': [([0.4901657998561859, 0.4901657998561859, 0.4901657998561859],
                            [0.25399115681648254, 0.25399115681648254, 0.25399115681648254]),
                           ([0.489375501871109, 0.489375501871109, 0.489375501871109],
                            [0.25438904762268066, 0.25438904762268066, 0.25438904762268066]),
                           ([0.48951974511146545, 0.48951974511146545, 0.48951974511146545],
                            [0.25359398126602173, 0.25359398126602173, 0.25359398126602173]),
                           ([0.4893500506877899, 0.4893500506877899, 0.4893500506877899],
                            [0.2509026825428009, 0.2509026825428009, 0.2509026825428009])],

    'MSMT17_hsv': [([0.44996383786201477, 0.28441840410232544, 0.346777081489563],
                    [0.2604794502258301, 0.18564878404140472, 0.2270435243844986]),
                   ([0.4601036310195923, 0.2872633635997772, 0.34665629267692566],
                    [0.2599087059497833, 0.1826837956905365, 0.2259720116853714]),
                   ([0.4545888304710388, 0.2885444760322571, 0.34607627987861633],
                    [0.2570095360279083, 0.18584665656089783, 0.2270125448703766]),
                   ([0.4539683759212494, 0.2874079644680023, 0.3452335596084595],
                    [0.2579506039619446, 0.18606449663639069, 0.22650931775569916])],
    'DukeMTMC_hsv': [([0.5074490308761597, 0.17174935340881348, 0.4815542697906494],
                      [0.2629970908164978, 0.13352371752262115, 0.20127098262310028]),
                     ([0.5192623138427734, 0.167582169175148, 0.47589024901390076],
                      [0.2650260329246521, 0.1318003237247467, 0.1984160989522934]),
                     ([0.5151113271713257, 0.1674039214849472, 0.4717757999897003],
                      [0.2633797526359558, 0.13170172274112701, 0.19604437053203583]),
                     ([0.522379994392395, 0.17204061150550842, 0.4649854600429535],
                      [0.2581104040145874, 0.13536538183689117, 0.1997414529323578])],
    'Market1501_hsv': [([0.39709705114364624, 0.15038226544857025, 0.4292905926704407],
                        [0.34003332257270813, 0.1375044733285904, 0.19681434333324432]),
                       ([0.38569408655166626, 0.14652986824512482, 0.42402273416519165],
                        [0.3351982533931732, 0.13509699702262878, 0.19481544196605682]),
                       ([0.38816311955451965, 0.1433437615633011, 0.4249427318572998],
                        [0.3436805307865143, 0.13522109389305115, 0.18959380686283112]),
                       ([0.3845953643321991, 0.1534813493490219, 0.43144291639328003],
                        [0.3316703140735626, 0.13693207502365112, 0.1899690330028534])],

    'MSMT17_hhh': [([0.44996383786201477, 0.44996383786201477, 0.44996383786201477],
                    [0.2604794502258301, 0.2604794502258301, 0.2604794502258301]),
                   ([0.4601036310195923, 0.4601036310195923, 0.4601036310195923],
                    [0.2599087059497833, 0.2599087059497833, 0.2599087059497833]),
                   ([0.4545888304710388, 0.4545888304710388, 0.4545888304710388],
                    [0.2570095360279083, 0.2570095360279083, 0.2570095360279083]),
                   ([0.4539683759212494, 0.4539683759212494, 0.4539683759212494],
                    [0.2579506039619446, 0.2579506039619446, 0.2579506039619446])],
    'DukeMTMC_hhh': [([0.5074490308761597, 0.5074490308761597, 0.5074490308761597],
                      [0.2629970908164978, 0.2629970908164978, 0.2629970908164978]),
                     ([0.5192623138427734, 0.5192623138427734, 0.5192623138427734],
                      [0.2650260329246521, 0.2650260329246521, 0.2650260329246521]),
                     ([0.5151113271713257, 0.5151113271713257, 0.5151113271713257],
                      [0.2633797526359558, 0.2633797526359558, 0.2633797526359558]),
                     ([0.522379994392395, 0.522379994392395, 0.522379994392395],
                      [0.2581104040145874, 0.2581104040145874, 0.2581104040145874])],
    'Market1501_hhh': [([0.39709705114364624, 0.39709705114364624, 0.39709705114364624],
                        [0.34003332257270813, 0.34003332257270813, 0.34003332257270813]),
                       ([0.38569408655166626, 0.38569408655166626, 0.38569408655166626],
                        [0.3351982533931732, 0.3351982533931732, 0.3351982533931732]),
                       ([0.38816311955451965, 0.38816311955451965, 0.38816311955451965],
                        [0.3436805307865143, 0.3436805307865143, 0.3436805307865143]),
                       ([0.3845953643321991, 0.3845953643321991, 0.3845953643321991],
                        [0.3316703140735626, 0.3316703140735626, 0.3316703140735626])],
    'mix': [([0.36877983808517456, 0.35007545351982117, 0.3589991331100464],
             [0.2126782387495041, 0.20730894804000854, 0.19884957373142242]),
            ([0.35252895951271057, 0.3328342139720917, 0.3442031145095825],
             [0.21349437534809113, 0.20689982175827026, 0.19968882203102112]),
            ([0.34611955285072327, 0.32659149169921875, 0.33623793721199036],
             [0.21464364230632782, 0.207486093044281, 0.1997326761484146]),
            ([0.3445225656032562, 0.3250943720340729, 0.33545082807540894],
             [0.2156081199645996, 0.2085423320531845, 0.20042452216148376])]
}

normalization = {k: mean_std[k][0] for k in mean_std.keys()}
