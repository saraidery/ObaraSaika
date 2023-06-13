import numpy as np
import pytest
import os

from obara_saika import OverlapIntegralPWGTO, NucAttIntegralPWGTO, KineticIntegralPWGTO

class TestPWGTO:

    def __overlap__(self, l_a, l_b, S_ref):


        k_a = np.array([0.25, 0.50, 0.75])
        k_b = np.array([-0.30, 0.00, -0.60])
        A = np.array([0.0, 1.2, 0.7])
        B = np.array([0.3, 0.0, -0.25])

        alpha = 0.8
        beta = 1.1

        S = OverlapIntegralPWGTO(A, alpha, l_a, k_a, B, beta, l_b, k_b)

        assert np.allclose(S.integral().flatten(), S_ref.flatten())

    def test_overlap_ss(self):
        S_ref = np.array([[0.4389493727424819 -0.24803857221744208j]])
        self.__overlap__(0, 0, S_ref)

    def test_overlap_ps(self):
        S_ref = np.array([[0.11213889492885043 +0.020451762459171886j],
                          [-0.2723176468240609 +0.23007803079612815j],
                          [-0.15330318856269487 +0.2923637550360012j],
                          ])
        self.__overlap__(1, 0, S_ref)


    def test_overlap_pp(self):
        S_ref = np.array([[0.09838795626949715 -0.05162605947025706j],
                         [0.05396863079837025 +0.02508863983843028j],
                         [0.03758980025578172 +0.04801952291891825j],
                         [0.001097250930968005 -0.06847688435141443j],
                         [-0.0523525064625734 +0.015145374605024976j],
                         [-0.1906653065124594 -0.004713214842728304j],
                         [-0.02295119335781772 -0.05911877792809539j],
                         [-0.11592736830483534 +0.1275491619441514j],
                         [-0.04967435346616212 -0.0027907813479890456j],])
        self.__overlap__(1, 1, S_ref)


    def test_overlap_sd(self):
        S_ref = np.array([[0.1042517313376654 -0.08008505970757843j],
                          [-0.03872493100780336 +0.07856429453395324j],
                          [-0.07191442807378395 +0.053695981702065715j],
                          [0.2529534140977276 -0.06593653243273735j],
                          [0.21784548973238116 +0.10974155942028221j],
                          [0.20083942629936766 +0.05109997450997082j],
                        ])
        self.__overlap__(0, 2, S_ref)


    def test_overlap_dp(self):
        S_ref = np.array([[0.048927200495122516 +0.03561980875827118j],
                          [0.07269530128492241 -0.005612370166403912j],
                          [0.06897295950159843 +0.02870904913068345j],
                          [-0.10662649480425915 +0.08454539964948651j],
                          [-0.019546015736163193 -0.008568150556563498j],
                          [-0.05617625752456686 -0.04921605978036357j],
                          [-0.06195985511770716 +0.1097218657933043j],
                          [-0.06684991488713 +0.009308585412267931j],
                          [-0.014243904306833054 -0.013292491783647867j],
                          [0.0031041533747519848 +0.07268182480990068j],
                          [0.029668916684615076 +0.025355192827045212j],
                          [0.20247677162024208 -0.006884909116314651j],
                          [0.04109087226848056 +0.06590816919064897j],
                          [0.040552999096369266 -0.046642175884251j],
                          [0.06041014163163159 -0.007962654154520476j],
                          [0.028482217743914143 +0.049325649726597365j],
                          [0.08539958726300498 -0.12911788198410695j],
                          [0.057363810669482145 +0.07575347955352918j],
                          ])
        self.__overlap__(2, 1, S_ref)


    def test_overlap_dd(self):
        S_ref = np.array([[0.07519199637695757 -0.03656057831701895j],
                          [0.03470041663851147 +0.042322950035062805j],
                          [0.011979685165275847 +0.05478468954825418j],
                          [0.07221336534168561 -0.005241744949878797j],
                          [0.0538182634021679 +0.04084351474387953j],
                          [0.052134554437936624 +0.02401597875612928j],
                          [-0.03500491147580441 -0.05707505209003816j],
                          [-0.034906580898491274 +0.008931439376677593j],
                          [-0.12589664795458783 -0.007036119347991577j],
                          [-0.020886255275110486 +0.004783566007603428j],
                          [-0.008269603956579864 -0.017963513746814607j],
                          [-0.041722667474163984 -0.039394670960973356j],
                          [-0.04993953557259028 -0.03674242278868896j],
                          [-0.07922940735450015 +0.08190141480614614j],
                          [-0.032767921478470415 -0.0028658724155191453j],
                          [-0.06642564098562397 +0.008938806026141798j],
                          [-0.009436053440612196 -0.014879016722487969j],
                          [-0.015265719264481484 +0.024541666178096055j],
                          [0.06130622959499296 -0.07740217207972841j],
                          [-0.012847457722699082 +0.0018904020118351561j],
                          [-0.04257296578057649 +0.05226553595760665j],
                          [0.05631852116370856 -0.0439844959394855j],
                          [0.004953320286996676 +0.035822881303130286j],
                          [0.1556547341831822 +0.0005079782842630987j],
                          [0.016281444796849218 -0.10415288582382015j],
                          [0.002820397316704906 +0.020370920261635516j],
                          [-0.01122068467103672 +0.01688642509929464j],
                          [0.004798251654522557 -0.06186838785951163j],
                          [0.015465524949957982 +0.004595859963811041j],
                          [-0.02890168362100927 -0.08564686719827137j],
                          [0.014516606263602799 -0.0759335370373169j],
                          [0.013684583923267398 +0.04965805907698671j],
                          [-0.03154112055158148 -0.0021931292584912447j],
                          [0.08539205900142681 -0.12782707589741382j],
                          [0.03293712334330631 +0.07936830477273424j],
                          [-0.004857693800941539 -0.024613546842895535j],])

        self.__overlap__(2, 2, S_ref)


    def __nuclear_attraction__(self, l_a, l_b, V_ref):


        k_a = np.array([0.25, 0.50, 0.75])
        k_b = np.array([-0.30, 0.00, -0.60])
        A = np.array([0.0, 1.2, 0.7])
        B = np.array([0.3, 0.0, -0.25])
        C = np.array([1.0, 0.0, 0.0])

        Z = 1.0
        alpha = 0.8
        beta = 1.1

        V = NucAttIntegralPWGTO(A, alpha, l_a, k_a, B, beta, l_b, k_b, C, Z)

        assert np.allclose(V.integral().flatten(), V_ref.flatten())


    def test_nuclear_attraction_ss(self):
        V_ref = np.array([[-0.4515895247979211935 +0.25526760149233762576j]])
        self.__nuclear_attraction__(0, 0, V_ref)

    def test_nuclear_attraction_ps(self):
        V_ref = np.array([[-0.18948799273008667954  +0.039897264164505827355j],
                          [0.33785095510809315389   -0.25210799421808444132j],
                          [0.19261728793700824691   -0.2739097524766610503j],])
        self.__nuclear_attraction__(1, 0, V_ref)


    def test_nuclear_attraction_ds(self):
        V_ref = np.array([[-0.17881743736552185853   +0.061782879948273757009j],
                         [0.27888367584135398403   -0.099278525564375857959j],
                         [0.18765227244153456776   -0.12309803492225876442j],
                         [-0.34760320176838943595   +0.29029807744514724011j],
                         [-0.2269116925035623844   +0.40810884754121490303j],
                         [-0.12792065343664332788   +0.26094900244612878604j],])
        self.__nuclear_attraction__(2, 0, V_ref)


    def test_nuclear_attraction_sd(self):
        V_ref = np.array([[-0.10576769895928275766      +0.060818605583880643028j],
                          [-0.0089288666460968606209      -0.044523586643220078307j],
                          [-0.0013071352684594363894      -0.041130358107303595372j],
                          [-0.18704982521797230977        +0.05282423747071079978j],
                          [-0.16232855017679995169        -0.071996985209108382842j],
                          [-0.16950735248645154574        -0.029100516912692518667j],])
        self.__nuclear_attraction__(0, 2, V_ref)


    def test_nuclear_attraction_pp(self):
        V_ref = np.array([[-0.12197103954649585189      +0.049813700698922006027j],
                          [-0.066372025956505958932   -0.009441766461934529564j],
                          [-0.071672503085417393898   -0.033168282642799529203j],
                          [0.059658278787170095081        +0.018313914806083811393j],
                          [0.057817944361322357039        -0.012231515616554049231j],
                          [0.18995088060348319647     -0.0038808415205918295188j],
                          [0.050555903627062498407        +0.011102242143918247125j],
                          [0.10013321877520453551     -0.093069949985404881732j],
                          [0.055065770103514530276        +0.00073473759330078963758j],])
        self.__nuclear_attraction__(1, 1, V_ref)


    def test_nuclear_attraction_pd(self):
        V_ref = np.array([[-0.059048638771829184413    -0.01479352628716103929j],
                          [-0.083036578876726316323    +0.011403134891633460307j],
                          [-0.096822720711566861995    -0.021969482229985216881j],
                          [-0.058520755424316998894    -0.0082285538793484026832j],
                          [-0.035927510852997932012    -0.038797819881835310385j],
                          [-0.047686636753288907942    -0.026692172459724539807j],
                          [0.08052657265258048036      -0.058687012281241360268j],
                          [0.0065476585304091016598    +0.011727751628908244727j],
                          [0.014339616522193373085     +0.032157705409661507812j],
                          [0.043932742398077762047     -0.037600315461770950076j],
                          [0.043483148994597736747     +0.02043751188291801843j],
                          [0.13505604928027728495      +0.0020771792910318637171j],
                          [0.044805092614557243125     -0.063337774786590056442j],
                          [0.02125347751383406536      +0.025098682991956274696j],
                          [0.0067247048351649057335    +0.0079627134830950324629j],
                          [0.090337823164232777806     -0.088264397159590640696j],
                          [0.035821645775301440284     +0.011510627433989032917j],
                          [0.0051942389034534131032    -0.051192924377875551056j],])
        self.__nuclear_attraction__(1, 2, V_ref)

    def test_nuclear_attraction_dp(self):
        V_ref = np.array([[-0.095639950635777931653    +0.00015058392251556191113j],
                          [-0.067852798954015317023    +0.0037510730607098318606j],
                          [-0.077402374792120295921    -0.022634571272278525278j],
                          [0.17047546617910203093      -0.092132697736641191999j],
                          [0.03659074369730936499      +0.0053720696784751087102j],
                          [0.11304098936649807638      +0.030141161002725132267j],
                          [0.10387431495763056744      -0.10393534972736931388j],
                          [0.073284224247053289525     -0.023261881617828373409j],
                          [0.035337718308323246008     +0.0083444943356433976672j],
                          [-0.067809642129510669384    -0.015205677207361316394j],
                          [-0.025448790835509063624    -0.02292249672190608753j],
                          [-0.20283604894693618714     +0.016456612811879073321j],
                          [-0.083824994931719451485    +0.0020231060265176580709j],
                          [-0.051653287367253067353    +0.040563838059289585025j],
                          [-0.078630408117331906936    +0.0099835040237955842424j],
                          [-0.044145598298906130186    -0.0059498552637775696528j],
                          [-0.074444921005258610225    +0.095062116333689702929j],
                          [-0.047118242694885369148    -0.051890925091511302947j],])
        self.__nuclear_attraction__(2, 1, V_ref)


    def test_nuclear_attraction_dd(self):
        V_ref = np.array([[-0.092643279216340593263        +0.028553004826418704187j],
                          [-0.051283409494897756742        -0.022692137203605574486j],
                          [-0.051165905708698666832        -0.040190433210639676942j],
                          [-0.060924390056485630551        +0.0045807756604335546705j],
                          [-0.045462184138144887124        -0.031676571380076290474j],
                          [-0.053049616142860182844        -0.018813910188952301383j],
                          [0.096357855132827727185     +0.0046346893118668101505j],
                          [0.042483797913857111739     -0.0025529453247304652146j],
                          [0.14116791542512582835      +0.010957218863578286477j],
                          [0.034748476410281489701     -0.011348457176485721165j],
                          [0.019436830301713000496     +0.019604606821437164932j],
                          [0.074397626283918619206     +0.0261678290952815501j],
                          [0.075042390851831550225     -0.0092576063467988342626j],
                          [0.076558064331874606245     -0.053468761010377952037j],
                          [0.0430859026688904731       +0.0037308711132953949544j],
                          [0.064401721790907126564     -0.021698999110231893123j],
                          [0.016305436851959596162     +0.013071994157130910688j],
                          [0.018980978418001913377     -0.024019234547844024036j],
                          [-0.083604708936338867353        +0.067542126794597701078j],
                          [0.0040631635889987281068        -0.0058840621030541283981j],
                          [-0.019030625318124310547        -0.033401775030755691265j],
                          [-0.049864217414531203376        +0.040113872186763238625j],
                          [-0.0048037786094437635517       -0.027826487271876677865j],
                          [-0.14406249696457310017     +0.006592216936309053428j],
                          [-0.055300860418996365675        +0.09324528430231041054j],
                          [-0.013931510080822578912        -0.010312444462841687323j],
                          [-0.016285072824599199082        -0.0094592924882698561917j],
                          [-0.024913054667767300238        +0.058567900910009594584j],
                          [-0.017534985649865341556        -0.0063743722567894192665j],
                          [0.00025700692167053063898       +0.067217768456887094786j],
                          [-0.029367623471573515298        +0.060035327319808907964j],
                          [-0.021523349014999478002        -0.019749817727090435843j],
                          [0.0098935031183656017145        -0.0049831885536307709422j],
                          [-0.066847963025220072386        +0.089701183166414039105j],
                          [-0.02297753373994724721     -0.050119822818016648847j],
                          [-0.0031564789222924688472       +0.023297462940626578048j],])
        self.__nuclear_attraction__(2, 2, V_ref)


    def __kinetic__(self, l_a, l_b, T_ref):


        k_a = np.array([0.25, 0.50, 0.75])
        k_b = np.array([-0.30, 0.00, -0.60])
        A = np.array([0.0, 1.2, 0.7])
        B = np.array([0.3, 0.0, -0.25])

        alpha = 0.8
        beta = 1.1

        T = KineticIntegralPWGTO(A, alpha, l_a, k_a, B, beta, l_b, k_b)

        assert np.allclose(T.integral().flatten(), T_ref.flatten())

    def test_kinetic_ss(self):
        V_ref = np.array([[0.05933909-0.30944365j]])
        self.__kinetic__(0, 0, V_ref)

    def test_kinetic_ps(self):
        V_ref = np.array([[0.12306986907572624 -0.0897442799774711j],
                          [-0.3245615556045744  +0.3088503016453532j],
                          [-0.17241069200444092  +0.2714997252507606j],])
        self.__kinetic__(1, 0, V_ref)

    def test_kinetic_pp(self):
        V_ref = np.array([[ 0.10678088-0.11427062j,  0.12398312-0.00591107j,  0.12109757+0.02397589j],
                          [ 0.02637412-0.11502188j, -0.2375066 +0.04594207j, -0.35803906+0.07266547j],
                          [-0.00184771-0.09464693j, -0.23022177+0.23264399j, -0.1219564 +0.00206026j],])
        self.__kinetic__(1, 1, V_ref)


    def test_kinetic_sd(self):
        V_ref = np.array([[-0.07071235-0.06058641j, -0.06475529+0.14010936j, -0.08281027+0.10631649j,
                            0.24768899-0.10443842j,  0.51440599+0.03257278j,  0.21448893+0.05130991j]])
        self.__kinetic__(0, 2, V_ref)


    def test_kinetic_dp(self):
        V_ref = np.array([[ 0.11344213-0.04309501j,  0.02668724-0.01253446j,  0.01870303-0.03237688j],
                          [-0.22710332+0.19082526j, -0.09727534-0.04053263j, -0.17914399-0.06568886j],
                          [-0.12762892+0.20220528j, -0.15752811+0.04995631j, -0.0510986 -0.02981375j],
                          [-0.01201529+0.11654787j,  0.03874334+0.04030681j,  0.35567431-0.12400021j],
                          [ 0.03221553+0.18602851j,  0.24694105-0.19374394j,  0.20084259-0.01274324j],
                          [ 0.04347449+0.07321511j,  0.11179003-0.29545321j,  0.00319762+0.04350689j],])
        self.__kinetic__(2, 1, V_ref)



    def test_kinetic_dd(self):
        V_ref = np.array([[ 0.04590804-0.05758706j,  0.14124287+0.02734381j,  0.13179599+0.06928649j,
                            0.00567452+0.01306222j,  0.07359924+0.01376209j, -0.00182181+0.01925483j,],
                          [-0.01479936-0.12796674j, -0.1902707 +0.03330161j, -0.3543885 +0.03260223j,
                           -0.01279368-0.01070628j, -0.05386266-0.09602576j, -0.10554453-0.09138287j,],
                          [-0.0374901 -0.10190947j, -0.23116768+0.22373799j, -0.11088933-0.00419899j,
                           -0.13880019+0.02424763j, -0.04666467-0.05151936j,  0.03674721+0.0186349j, ],
                          [ 0.00537056-0.07288893j, -0.02493451-0.00384593j, -0.08767286+0.12040942j,
                           -0.10381087+0.02456278j,  0.01772305+0.07196652j,  0.28352034+0.04173934j,],
                          [ 0.04840921-0.12802706j, -0.01304916+0.11429188j, -0.05288281+0.05556289j,
                            0.10576851-0.02419191j,  0.09671448+0.04050391j, -0.05111797-0.00274436j,],
                          [-0.0432116 -0.04869429j,  0.03403675+0.11998052j, -0.02424969-0.02329895j,
                            0.12765808-0.22478289j,  0.02339326+0.11232354j, -0.1105277 +0.01052904j,],])
        self.__kinetic__(2, 2, V_ref)