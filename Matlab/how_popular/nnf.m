function [Y,Xf,Af] = nnf(X,~,~)
%MYNEURALNETWORKFUNCTION neural network simulation function.
%
% Generated by Neural Network Toolbox function genFunction, 28-Feb-2016 21:15:19.
%
% [Y] = myNeuralNetworkFunction(X,~,~) takes these arguments:
%
%   X = 1xTS cell, 1 inputs over TS timesteps
%   Each X{1,ts} = Qx61 matrix, input #1 at timestep ts.
%
% and returns:
%   Y = 1xTS cell of 1 outputs over TS timesteps.
%   Each Y{1,ts} = Qx1 matrix, output #1 at timestep ts.
%
% where Q is number of samples (or series) and TS is the number of timesteps.

%#ok<*RPMT0>

% ===== NEURAL NETWORK CONSTANTS =====

% Input 1
x1_step1_xoffset = [0;0;0;0;-20;0;0;0;0;-1256;0;0;0;0;-20;0;-0.5;0;-0.056;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;-1256;0;-138;0;-0.667;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0];
x1_step1_gain = [0.0147058823529412;0.00112485939257593;0.00112485939257593;0.00112485939257593;0.0465116279069767;0.0769230769230769;0.0833333333333333;0.0833333333333333;0.0666666666666667;0.000746825989544436;0.0010351966873706;0.00140449438202247;0.00140449438202247;0.000978473581213307;0.1;0.257135510413988;1.33333333333333;0.0869565217391304;2.76625172890733;0.25;0.403388463089956;0.181818181818182;0.0769230769230769;0.235543516664704;0.417798203467725;2;0.0833333333333333;0.666666666666667;0.419903422212891;2;0.0833333333333333;0.634115409004439;0.25;0.385059684251059;0.181818181818182;0.0666666666666667;0.223388808220708;0.00159235668789809;0.00370216113656347;0.0060790273556231;0.00140646976090014;0.0637836458731981;0.00275482093663912;0.00344105557820917;0.00155038759689922;0.0010351966873706;0.00181488203266788;0.0054983724817454;0.00340136054421769;0.00140449438202247;0.0045662100456621;0.142857142857143;0.00556281812366145;0.00340136054421769;0.00140449438202247;0.00451806888699632;0.00275482093663912;0.00357504830784026;0.0015220700152207;0.000978473581213307;0.00178147215514485];
x1_step1_ymin = -1;

% Layer 1
b1 = [-6.7121366964796865;-0.65336843457827676;-45.686443028734324;60.280339932795819;40.595011053785896;14.657886253964804;44.211445165380141;-27.818245816124172;22.290365971078629;5.4472479210152791];
IW1_1 = [-0.036613285984276954 7.2175824464631981 -6.0143462227268261 -5.7106111771762498 -0.89460558997994699 1.057962471808261 1.7467586793035457 0.25192787169008229 -0.79850748629765589 -0.036413526524328894 -3.8153579159516897 4.0080039977252682 3.5225952033777794 12.092625128780345 13.953876292571714 -6.3723627734245571 -3.0080526907413541 -2.1900128397308243 0.89201057805196626 -10.814600626068426 -12.252143677509313 0.86384050424518 2.6256000212100754 11.968337436122759 29.389670591209345 21.03116464355746 19.265948600479106 -4.0458362988935406 7.8293296793424654 -11.175002653178707 -5.5488125367954027 -11.153295891768581 -10.650599292331451 -6.6191752754825259 4.3816763368723297 2.7809965733111448 -3.6099351432507172 3.5614635134675403 -14.131224958190785 -8.6012374145495176 -1.7784687803188926 4.9632591250759566 -14.143841039057664 35.59172338094973 4.8351490176114051 -5.8223867425591838 3.9977107320517957 45.759907658063049 48.975767563625404 -23.875495175993976 -19.693291578501203 -25.241390149763749 -19.554028002801584 -16.707752701749428 31.439160791829664 -19.817743673497521 -14.410289488606152 -34.578834731109652 2.5832487980634804 -2.931740587343278 19.705303265476697;-3.8431024029583232 -15.217302088826132 4.4809956516967189 4.6917434926383539 -24.332182439742677 6.0206341869818862 -10.384936763986477 -55.040139977859745 -27.301855283628317 17.632214513641028 -54.9099445488634 39.95703739754606 72.328603271680805 138.93715357303535 11.986153039275628 -11.680481736940513 4.4158106265670138 -40.092482967236123 -48.352838779778835 3.263260045244686 -6.3067864704552319 32.15281553569379 -10.916457937661731 7.2321854196337982 -24.633352962740688 -1.3424042280153941 23.215136886000575 7.145979128254953 -20.977805255462144 -9.2067276050740983 23.578272119248386 -2.6365794537713438 3.3805263352190251 -22.157466957272739 22.097464499862777 13.672040794649259 1.5117021233783288 72.203751049055782 8.0925582007576295 -6.0719681755249937 7.3669971312666789 -61.845885847931463 1.954616801511589 42.398189060118817 -0.86718845774758158 -32.285495145590289 -33.653505456001071 19.423086528443129 -0.34420794704279223 -6.0873476063663539 -19.46306152898882 0.35591414028012103 22.164886957427747 -2.8615103628057645 13.307930707687378 -23.649446852777015 1.6906572495995396 41.29869640302045 -4.320000344562855 51.486501042434028 -34.763213399325522;2.2846019953816454 -12.351677128877606 20.452742622965623 20.811609546695312 -18.82407468081491 -1.098516766254712 21.358662083126518 -18.16103848273956 -11.683418953507459 -12.765666219718305 19.279287650029342 -19.158448446188149 -49.052982562605081 -15.997544654108561 13.053605869311536 16.117908845388349 15.146215477940281 36.528065539154191 -8.365918944549021 24.211062566883392 -17.065186396565391 18.346647730156842 -1.6673977933927309 29.288649606525311 14.237789500289992 45.598432866528235 0.63203263792813213 -9.6756203477366842 16.821148038009742 98.290033377042292 21.397506876560513 -19.582086069932501 24.327114006507571 -27.828344226593003 -16.794336344863236 -19.419405693226725 13.81224415995527 -1.7815567656852138 35.469726925160941 4.5771271874544404 25.790231587013352 13.651447004314123 0.84545795290631198 21.21736953751104 42.383212402702767 21.458039901435939 -23.811702643769966 14.474783173480741 -25.858087278423888 -33.698303508728877 -50.053231320707269 -122.45498740615069 2.7058241311634208 -43.735679758186009 -42.372019488751448 -51.247236816232103 1.3489131996876162 -18.318715656447132 4.5997906622046809 44.137860483233226 -37.934696380300977;-4.3962459516993109 6.7248531722239671 5.0707458847010116 5.4699056218980289 3.2383517426880384 7.8845538826265411 -4.3462241716045504 9.6158693195151965 -1.9261552935096693 -15.312929302699018 3.3848798775392224 16.22035443971847 -4.9620129974927965 40.038715765551494 -19.126890391767986 -9.0478787218214762 -9.87486935529585 -6.4131407869935355 16.119074083523486 -11.526097415005328 5.6599511187206915 -6.2539459360211476 -2.3324346521926089 10.599973799374697 9.6099656535116651 17.27515813948618 -32.898551896453121 -6.4517450485204897 -27.459726065753348 19.920104306348946 20.34865991989167 7.5291177441164434 -11.23011116575924 14.761908530662001 13.114390488991498 -12.844866764815164 11.682294472579278 5.3529712053613032 -25.989223614433989 -29.917911148090447 6.7432207029986824 -42.821786240113795 19.22020889365298 -6.9666423865439526 -15.65646958355361 -16.090101521041895 12.312036831826571 1.4829480065880902 -27.703734608267137 -2.0281723915648215 -4.2485196784641541 98.885131232732945 10.768655799142335 -55.162661026572707 18.906606835901307 -3.9520011876954091 19.600592824808462 13.199888674554789 -34.617067030969487 -13.130921757590134 11.009732633979397;-40.216262506646117 -40.433584820606136 -40.780468453969029 -40.436925131148072 -15.858938570291009 -53.28688574141006 5.5418738443551501 -18.645388270526585 47.249260748376088 -10.905072552571662 103.53617281473306 -18.236604051864525 -33.983826917515664 -31.011550900835058 31.324735048878683 -3.3517698267092566 -13.610969739173857 -35.613897148225981 -21.014444409369133 -41.064015822299567 -20.643298926655731 -3.2880738133705503 -84.287281694896564 -8.4272664957549033 -7.5490948284135664 -40.534782431566533 -35.55178355394029 -10.96119245594751 -5.0507952088123202 -28.257151979740922 -35.342102634984592 -9.445218004997356 -41.134408001198828 -19.602996652960794 -8.7255186148298307 -72.617730184308442 -8.9132874218234299 -27.031717483711773 77.513043669721341 56.649872944278385 -46.178122680958168 -19.474580126788616 -9.536200110355578 60.371535925842608 6.5070372036597499 -18.585935999497423 37.589317182944605 76.504189867665588 -4.3913180109430492 -22.666808634382022 43.704538696318707 -36.675069116692129 76.721549979533265 -3.8125971213819327 -22.203171729745669 44.476468436916925 -9.5527924556739858 62.117475731704673 18.405758793593805 -17.04601089147415 42.001136809828637;-4.5390276768161639 5.4542853302040033 -2.1468010175468222 -1.816793359013293 8.4471913507041059 12.41677514996775 3.2766667339386935 20.236041429145807 -16.214438862243014 20.364724539567938 -37.99224850150501 -7.5203153672408218 32.619842968042015 29.804877415676184 -13.129646333686223 -27.307803285868271 -1.5464273456569699 25.812531375675285 -3.2066229802241342 -8.4251081007853514 -28.341141584852195 -8.7414413425465902 -14.723361944697684 31.963724163397721 8.384257132376332 22.258462721634352 -6.4276127147954902 4.0426814528305703 1.4729486685646855 -9.3555804902526312 -33.60332573782118 4.3774731520802037 -8.5171580056493514 4.9311485664850672 -18.066686127027594 16.46038285195322 25.502685148385993 0.30632930342936554 39.509653458508119 7.0238444384925272 -8.2232559240328875 2.1993682019848713 -9.1949214925325204 -13.513617747446721 11.853150911724036 21.248055682550831 -28.962280355978578 9.7659832650736842 27.399520772002976 -10.978061092710281 -8.9768512607105091 -45.359449879239811 -2.6785886301131856 58.3805796717281 0.1017837867124388 -8.0081236401348974 -8.8418772395328986 0.96478423728582641 -5.0844435997036364 1.9826165272622505 -28.872228699626273;4.3256349765907895 1.0000289901319863 -0.60957653125746247 -0.37356500992591973 -1.9550284349635034 1.8860323836069131 -0.97258087191281917 0.9474613299229423 -1.9730745294742993 -1.6263315194735668 -0.14097891757637393 -0.82171953925548102 1.1422298158933031 3.0352805213465159 -1.7478911012283607 -14.491582373659204 8.4474724284958835 0.31180185823795431 1.3467746568705607 -9.4392097005344962 -2.4620689024287934 0.63923641305672108 3.5199606571616875 3.9996658007782604 10.411701998915467 2.2129833260280938 0.48867686809521399 -13.373532594011776 10.900209883227076 2.5640681881364973 -2.6874730939201052 -0.13465663801634198 -9.5694550619817971 -2.4957575243441741 1.9902743853389755 -4.2710113165903598 10.185806549852785 -3.0447782551587519 -18.10065145490541 13.310549039675561 1.9363119156636623 -2.1890034187389094 -6.1331067340066534 6.7728102531222074 2.1500315581302347 1.3316090310741762 1.0400131999278976 -16.641188333415812 -3.4000257552574951 8.1213277966284103 11.713548699926729 28.704162312525167 46.862906291484357 39.545648405059126 -18.190208428530507 13.803261845173258 -5.8594148446644949 -15.918565245947796 -9.4525091904190219 4.2555064255695028 -29.662842672776605;-2.6103271565004245 -5.1734643394514075 0.66937441357986149 0.84380278012933474 0.70914445874200494 1.4000963680943477 1.5401984615522144 -0.18082372776623445 -4.3624928939825187 1.9565144853129912 -5.2679828064650067 2.6920697188604579 2.4057687405674404 5.4965884593374827 4.9282145414832321 -4.8503975752899082 -8.9151285859828473 31.269379822025254 4.644107278478244 10.753513397141925 7.9516944121624826 -14.431388098216617 -5.636692105900929 0.78452241518224586 21.740497517880133 32.58387905288086 -7.3522719247858372 12.083538245621273 13.803249725143553 -30.145833905841634 -7.281198340889758 8.1631939539763376 10.646974847262314 -29.120852909045642 -1.817681025199831 -2.9004158878790292 -19.667742459205023 16.081312883910083 32.138890902751449 -9.4007893683781916 4.0534938010944384 -8.0852615242604724 -31.532335866290602 -22.59692006397032 -31.793964876567923 20.156060103132042 -2.9063356575178472 -5.4988791324694262 30.919279825359151 -30.372535028161106 -5.2510385704983982 12.532657636647095 -5.5427643029241329 23.555696924606952 24.592673712754902 -7.5183550992342827 -31.016131355194538 13.680500560172302 -4.1389997043702271 -6.9139470197795934 -1.1935700688796766;1.5868384934273601 0.34510833954222397 -1.2854834429362025 -1.1797998638530649 -0.82069388891546902 0.83939016320238591 -0.33352670128912165 0.48286715782635614 -0.59700511573477 -1.5960799703967108 -0.064478013532376005 -0.53007439170872495 -1.4359382510494336 3.7537775813751564 0.90596307672033882 -11.128465252860313 -3.4527704652895754 1.9716786093873082 -1.8793483123250452 -2.7024699577120326 -1.4094027589554032 0.091051305090346216 0.89390189943964904 1.2606195767001429 15.941404067155556 -5.2225631646303423 -0.20645874891834279 -22.258500303154349 3.4583213692234605 1.5061333543880402 -0.41974283286217234 17.539418863008141 -2.8210894982575425 -5.5374208341340774 1.015908009239421 0.58477005821533878 3.9976413993717701 0.56823948865457685 -1.4273580583413064 -0.36394795526570051 0.68917294856317168 2.539118453167958 -5.4817038718110416 1.6513359895044852 -4.641726940914034 3.8353876848610495 22.696665627783776 15.509999784052738 34.190468902992258 -7.7924751192090227 -7.991578528935988 31.872129204407045 -9.4964360168523001 -23.667984691540426 2.5192594089373563 -5.166413923000488 -5.2824965696990738 -4.1179398013090021 -0.1113534272428524 0.24661258782938905 -9.83124194925451;-7.5223531096639444 -4.5766559105413389 -4.6324727095915836 -4.4069771648010931 -15.895058291741433 27.049373390849908 32.237989314601755 4.5705969330845901 20.619591843265926 1.8688697387007642 3.9553373158460063 1.8419335174523019 5.3373906557631319 4.4229341861373666 -10.552349735259678 -4.0804155056660862 -1.3379015837158135 11.65251694727972 -7.7664419195369767 -6.3317836149637667 -0.54932565539830525 -7.271643178090974 22.79154511308208 -8.3091563654765377 -3.8947993343384861 -5.8125649038681306 11.703348631989478 -13.491728798256622 -4.6031029347247676 3.6603487756534907 12.879483126020443 -8.9915241949315892 -6.6936605280244876 -1.7349175851225385 -6.2089567472286964 15.249530675736734 -9.5948837649813683 -9.6063510287227736 0.20098613475082555 -0.84564502423146004 17.073569335380167 3.7592963498618914 -8.3628654232225905 -9.365109134481628 -6.6277916464409206 -3.6527321407850661 -6.8977859500481467 1.592597890634547 -6.2390272248843983 15.576328514923528 -8.3568878001272591 -5.8599507795564616 -1.1169073038303534 -2.5730157722913889 16.88348127352954 -7.6550041412399441 -8.4348153889603381 -4.3740776484448398 -6.8822704682730311 -16.986203484885564 -9.3849931224616849];

% Layer 2
b2 = -0.89321629025841054;
LW2_1 = [0.0087565367936877376 -0.0015104159281223218 -0.00012711301763198218 -0.0069444947200863428 -0.069491755769023208 0.018795939912299404 0.11518475195234423 0.0010316391031956638 -0.10459206652206783 0.0025681378355564143];

% Output 1
y1_step1_ymin = -1;
y1_step1_gain = 0.00140449438202247;
y1_step1_xoffset = 0;

% ===== SIMULATION ========

% Format Input Arguments
isCellX = iscell(X);
if ~isCellX, X = {X}; end;

% Dimensions
TS = size(X,2); % timesteps
if ~isempty(X)
    Q = size(X{1},1); % samples/series
else
    Q = 0;
end

% Allocate Outputs
Y = cell(1,TS);

% Time loop
for ts=1:TS
    
    % Input 1
    X{1,ts} = X{1,ts}';
    Xp1 = mapminmax_apply(X{1,ts},x1_step1_gain,x1_step1_xoffset,x1_step1_ymin);
    
    % Layer 1
    a1 = tansig_apply(repmat(b1,1,Q) + IW1_1*Xp1);
    
    % Layer 2
    a2 = repmat(b2,1,Q) + LW2_1*a1;
    
    % Output 1
    Y{1,ts} = mapminmax_reverse(a2,y1_step1_gain,y1_step1_xoffset,y1_step1_ymin);
    Y{1,ts} = Y{1,ts}';
end

% Final Delay States
Xf = cell(1,0);
Af = cell(2,0);

% Format Output Arguments
if ~isCellX, Y = cell2mat(Y); end
end

% ===== MODULE FUNCTIONS ========

% Map Minimum and Maximum Input Processing Function
function y = mapminmax_apply(x,settings_gain,settings_xoffset,settings_ymin)
y = bsxfun(@minus,x,settings_xoffset);
y = bsxfun(@times,y,settings_gain);
y = bsxfun(@plus,y,settings_ymin);
end

% Sigmoid Symmetric Transfer Function
function a = tansig_apply(n)
a = 2 ./ (1 + exp(-2*n)) - 1;
end

% Map Minimum and Maximum Output Reverse-Processing Function
function x = mapminmax_reverse(y,settings_gain,settings_xoffset,settings_ymin)
x = bsxfun(@minus,y,settings_ymin);
x = bsxfun(@rdivide,x,settings_gain);
x = bsxfun(@plus,x,settings_xoffset);
end