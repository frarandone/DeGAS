# Generated from ASGMT.g4 by ANTLR 4.10.1
# encoding: utf-8
from antlr4 import *
import torch
import numpy as np
from io import StringIO
import sys
if sys.version_info[1] > 5:
	from typing import TextIO
else:
	from typing.io import TextIO

def serializedATN():
    return [
        4,1,19,308,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,6,7,
        6,2,7,7,7,2,8,7,8,2,9,7,9,2,10,7,10,2,11,7,11,2,12,7,12,2,13,7,13,
        2,14,7,14,2,15,7,15,2,16,7,16,2,17,7,17,2,18,7,18,2,19,7,19,2,20,
        7,20,2,21,7,21,1,0,1,0,1,0,1,0,1,0,1,0,3,0,51,8,0,1,1,1,1,3,1,55,
        8,1,1,1,5,1,58,8,1,10,1,12,1,61,9,1,1,2,1,2,1,2,3,2,66,8,2,1,2,1,
        2,1,3,1,3,3,3,72,8,3,1,3,1,3,3,3,76,8,3,1,3,1,3,3,3,80,8,3,1,3,3,
        3,83,8,3,1,4,1,4,3,4,87,8,4,1,4,5,4,90,8,4,10,4,12,4,93,9,4,1,5,
        3,5,96,8,5,1,5,1,5,1,5,3,5,101,8,5,1,5,1,5,1,5,3,5,106,8,5,1,5,1,
        5,3,5,110,8,5,1,5,1,5,1,5,3,5,115,8,5,3,5,117,8,5,1,6,1,6,1,6,5,
        6,122,8,6,10,6,12,6,125,9,6,1,7,1,7,3,7,129,8,7,1,7,1,7,3,7,133,
        8,7,1,8,1,8,3,8,137,8,8,1,8,5,8,140,8,8,10,8,12,8,143,9,8,1,9,3,
        9,146,8,9,1,9,1,9,1,9,3,9,151,8,9,1,9,1,9,1,9,3,9,156,8,9,1,9,1,
        9,3,9,160,8,9,1,9,1,9,1,9,3,9,165,8,9,3,9,167,8,9,1,10,1,10,1,10,
        5,10,172,8,10,10,10,12,10,175,9,10,1,11,1,11,3,11,179,8,11,1,11,
        1,11,3,11,183,8,11,1,11,1,11,1,11,3,11,188,8,11,1,11,1,11,1,11,1,
        11,1,11,3,11,195,8,11,1,11,1,11,3,11,199,8,11,1,11,1,11,3,11,203,
        8,11,3,11,205,8,11,1,12,1,12,3,12,209,8,12,1,12,5,12,212,8,12,10,
        12,12,12,215,9,12,1,13,3,13,218,8,13,1,13,1,13,1,13,3,13,223,8,13,
        1,13,1,13,1,13,3,13,228,8,13,1,13,1,13,3,13,232,8,13,1,13,1,13,1,
        13,3,13,237,8,13,3,13,239,8,13,1,14,1,14,1,14,5,14,244,8,14,10,14,
        12,14,247,9,14,1,15,1,15,3,15,251,8,15,1,15,1,15,3,15,255,8,15,1,
        15,1,15,1,15,3,15,260,8,15,1,15,1,15,1,15,3,15,265,8,15,3,15,267,
        8,15,1,16,1,16,3,16,271,8,16,1,17,1,17,1,17,1,17,1,17,1,18,1,18,
        1,18,1,18,1,18,1,18,1,18,1,18,1,19,1,19,1,19,3,19,289,8,19,1,19,
        1,19,1,19,3,19,294,8,19,5,19,296,8,19,10,19,12,19,299,9,19,1,19,
        1,19,1,20,1,20,1,21,1,21,1,21,1,21,5,59,91,141,213,297,0,22,0,2,
        4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,0,1,1,0,
        15,16,351,0,44,1,0,0,0,2,52,1,0,0,0,4,65,1,0,0,0,6,82,1,0,0,0,8,
        84,1,0,0,0,10,116,1,0,0,0,12,118,1,0,0,0,14,128,1,0,0,0,16,134,1,
        0,0,0,18,166,1,0,0,0,20,168,1,0,0,0,22,204,1,0,0,0,24,206,1,0,0,
        0,26,238,1,0,0,0,28,240,1,0,0,0,30,266,1,0,0,0,32,270,1,0,0,0,34,
        272,1,0,0,0,36,277,1,0,0,0,38,285,1,0,0,0,40,302,1,0,0,0,42,304,
        1,0,0,0,44,45,3,32,16,0,45,50,5,1,0,0,46,51,3,2,1,0,47,51,3,8,4,
        0,48,51,3,16,8,0,49,51,3,24,12,0,50,46,1,0,0,0,50,47,1,0,0,0,50,
        48,1,0,0,0,50,49,1,0,0,0,51,1,1,0,0,0,52,59,3,4,2,0,53,55,5,2,0,
        0,54,53,1,0,0,0,54,55,1,0,0,0,55,56,1,0,0,0,56,58,3,4,2,0,57,54,
        1,0,0,0,58,61,1,0,0,0,59,60,1,0,0,0,59,57,1,0,0,0,60,3,1,0,0,0,61,
        59,1,0,0,0,62,63,3,6,3,0,63,64,5,3,0,0,64,66,1,0,0,0,65,62,1,0,0,
        0,65,66,1,0,0,0,66,67,1,0,0,0,67,68,3,6,3,0,68,5,1,0,0,0,69,83,5,
        16,0,0,70,72,3,40,20,0,71,70,1,0,0,0,71,72,1,0,0,0,72,73,1,0,0,0,
        73,83,3,42,21,0,74,76,3,40,20,0,75,74,1,0,0,0,75,76,1,0,0,0,76,77,
        1,0,0,0,77,83,3,32,16,0,78,80,3,40,20,0,79,78,1,0,0,0,79,80,1,0,
        0,0,80,81,1,0,0,0,81,83,3,36,18,0,82,69,1,0,0,0,82,71,1,0,0,0,82,
        75,1,0,0,0,82,79,1,0,0,0,83,7,1,0,0,0,84,91,3,10,5,0,85,87,5,2,0,
        0,86,85,1,0,0,0,86,87,1,0,0,0,87,88,1,0,0,0,88,90,3,10,5,0,89,86,
        1,0,0,0,90,93,1,0,0,0,91,92,1,0,0,0,91,89,1,0,0,0,92,9,1,0,0,0,93,
        91,1,0,0,0,94,96,3,40,20,0,95,94,1,0,0,0,95,96,1,0,0,0,96,100,1,
        0,0,0,97,101,5,16,0,0,98,101,3,42,21,0,99,101,3,34,17,0,100,97,1,
        0,0,0,100,98,1,0,0,0,100,99,1,0,0,0,101,102,1,0,0,0,102,103,5,3,
        0,0,103,117,3,12,6,0,104,106,3,40,20,0,105,104,1,0,0,0,105,106,1,
        0,0,0,106,107,1,0,0,0,107,117,3,12,6,0,108,110,3,40,20,0,109,108,
        1,0,0,0,109,110,1,0,0,0,110,114,1,0,0,0,111,115,5,16,0,0,112,115,
        3,42,21,0,113,115,3,34,17,0,114,111,1,0,0,0,114,112,1,0,0,0,114,
        113,1,0,0,0,115,117,1,0,0,0,116,95,1,0,0,0,116,105,1,0,0,0,116,109,
        1,0,0,0,117,11,1,0,0,0,118,123,3,14,7,0,119,120,5,3,0,0,120,122,
        3,14,7,0,121,119,1,0,0,0,122,125,1,0,0,0,123,121,1,0,0,0,123,124,
        1,0,0,0,124,13,1,0,0,0,125,123,1,0,0,0,126,129,3,32,16,0,127,129,
        3,36,18,0,128,126,1,0,0,0,128,127,1,0,0,0,129,132,1,0,0,0,130,131,
        5,4,0,0,131,133,5,16,0,0,132,130,1,0,0,0,132,133,1,0,0,0,133,15,
        1,0,0,0,134,141,3,18,9,0,135,137,5,2,0,0,136,135,1,0,0,0,136,137,
        1,0,0,0,137,138,1,0,0,0,138,140,3,18,9,0,139,136,1,0,0,0,140,143,
        1,0,0,0,141,142,1,0,0,0,141,139,1,0,0,0,142,17,1,0,0,0,143,141,1,
        0,0,0,144,146,3,40,20,0,145,144,1,0,0,0,145,146,1,0,0,0,146,150,
        1,0,0,0,147,151,5,16,0,0,148,151,3,42,21,0,149,151,3,34,17,0,150,
        147,1,0,0,0,150,148,1,0,0,0,150,149,1,0,0,0,151,152,1,0,0,0,152,
        153,5,3,0,0,153,167,3,20,10,0,154,156,3,40,20,0,155,154,1,0,0,0,
        155,156,1,0,0,0,156,157,1,0,0,0,157,167,3,20,10,0,158,160,3,40,20,
        0,159,158,1,0,0,0,159,160,1,0,0,0,160,164,1,0,0,0,161,165,5,16,0,
        0,162,165,3,42,21,0,163,165,3,34,17,0,164,161,1,0,0,0,164,162,1,
        0,0,0,164,163,1,0,0,0,165,167,1,0,0,0,166,145,1,0,0,0,166,155,1,
        0,0,0,166,159,1,0,0,0,167,19,1,0,0,0,168,173,3,22,11,0,169,170,5,
        3,0,0,170,172,3,22,11,0,171,169,1,0,0,0,172,175,1,0,0,0,173,171,
        1,0,0,0,173,174,1,0,0,0,174,21,1,0,0,0,175,173,1,0,0,0,176,179,3,
        32,16,0,177,179,3,36,18,0,178,176,1,0,0,0,178,177,1,0,0,0,179,182,
        1,0,0,0,180,181,5,4,0,0,181,183,5,16,0,0,182,180,1,0,0,0,182,183,
        1,0,0,0,183,205,1,0,0,0,184,187,5,5,0,0,185,188,3,32,16,0,186,188,
        3,36,18,0,187,185,1,0,0,0,187,186,1,0,0,0,188,189,1,0,0,0,189,190,
        5,6,0,0,190,199,1,0,0,0,191,194,5,7,0,0,192,195,3,32,16,0,193,195,
        3,36,18,0,194,192,1,0,0,0,194,193,1,0,0,0,195,196,1,0,0,0,196,197,
        5,6,0,0,197,199,1,0,0,0,198,184,1,0,0,0,198,191,1,0,0,0,199,202,
        1,0,0,0,200,201,5,4,0,0,201,203,5,16,0,0,202,200,1,0,0,0,202,203,
        1,0,0,0,203,205,1,0,0,0,204,178,1,0,0,0,204,198,1,0,0,0,205,23,1,
        0,0,0,206,213,3,26,13,0,207,209,5,2,0,0,208,207,1,0,0,0,208,209,
        1,0,0,0,209,210,1,0,0,0,210,212,3,26,13,0,211,208,1,0,0,0,212,215,
        1,0,0,0,213,214,1,0,0,0,213,211,1,0,0,0,214,25,1,0,0,0,215,213,1,
        0,0,0,216,218,3,40,20,0,217,216,1,0,0,0,217,218,1,0,0,0,218,222,
        1,0,0,0,219,223,5,16,0,0,220,223,3,42,21,0,221,223,3,34,17,0,222,
        219,1,0,0,0,222,220,1,0,0,0,222,221,1,0,0,0,223,224,1,0,0,0,224,
        225,5,3,0,0,225,239,3,28,14,0,226,228,3,40,20,0,227,226,1,0,0,0,
        227,228,1,0,0,0,228,229,1,0,0,0,229,239,3,28,14,0,230,232,3,40,20,
        0,231,230,1,0,0,0,231,232,1,0,0,0,232,236,1,0,0,0,233,237,5,16,0,
        0,234,237,3,42,21,0,235,237,3,34,17,0,236,233,1,0,0,0,236,234,1,
        0,0,0,236,235,1,0,0,0,237,239,1,0,0,0,238,217,1,0,0,0,238,227,1,
        0,0,0,238,231,1,0,0,0,239,27,1,0,0,0,240,245,3,30,15,0,241,242,5,
        3,0,0,242,244,3,30,15,0,243,241,1,0,0,0,244,247,1,0,0,0,245,243,
        1,0,0,0,245,246,1,0,0,0,246,29,1,0,0,0,247,245,1,0,0,0,248,251,3,
        32,16,0,249,251,3,36,18,0,250,248,1,0,0,0,250,249,1,0,0,0,251,254,
        1,0,0,0,252,253,5,4,0,0,253,255,5,16,0,0,254,252,1,0,0,0,254,255,
        1,0,0,0,255,267,1,0,0,0,256,259,5,8,0,0,257,260,3,32,16,0,258,260,
        3,36,18,0,259,257,1,0,0,0,259,258,1,0,0,0,260,261,1,0,0,0,261,264,
        5,6,0,0,262,263,5,4,0,0,263,265,5,16,0,0,264,262,1,0,0,0,264,265,
        1,0,0,0,265,267,1,0,0,0,266,250,1,0,0,0,266,256,1,0,0,0,267,31,1,
        0,0,0,268,271,5,15,0,0,269,271,3,34,17,0,270,268,1,0,0,0,270,269,
        1,0,0,0,271,33,1,0,0,0,272,273,5,15,0,0,273,274,5,9,0,0,274,275,
        7,0,0,0,275,276,5,10,0,0,276,35,1,0,0,0,277,278,5,11,0,0,278,279,
        3,38,19,0,279,280,5,12,0,0,280,281,3,38,19,0,281,282,5,12,0,0,282,
        283,3,38,19,0,283,284,5,6,0,0,284,37,1,0,0,0,285,288,5,9,0,0,286,
        289,5,16,0,0,287,289,3,42,21,0,288,286,1,0,0,0,288,287,1,0,0,0,289,
        297,1,0,0,0,290,293,5,12,0,0,291,294,5,16,0,0,292,294,3,42,21,0,
        293,291,1,0,0,0,293,292,1,0,0,0,294,296,1,0,0,0,295,290,1,0,0,0,
        296,299,1,0,0,0,297,298,1,0,0,0,297,295,1,0,0,0,298,300,1,0,0,0,
        299,297,1,0,0,0,300,301,5,10,0,0,301,39,1,0,0,0,302,303,5,13,0,0,
        303,41,1,0,0,0,304,305,5,14,0,0,305,306,5,15,0,0,306,43,1,0,0,0,
        53,50,54,59,65,71,75,79,82,86,91,95,100,105,109,114,116,123,128,
        132,136,141,145,150,155,159,164,166,173,178,182,187,194,198,202,
        204,208,213,217,222,227,231,236,238,245,250,254,259,264,266,270,
        288,293,297
    ]

class ASGMTParser ( Parser ):

    grammarFileName = "ASGMT.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    sharedContextCache = PredictionContextCache()

    literalNames = [ "<INVALID>", "'='", "'+'", "'*'", "'^'", "'cos('", 
                     "')'", "'sin('", "'exp('", "'['", "']'", "'gm('", "','", 
                     "'-'", "'_'" ]

    symbolicNames = [ "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "IDV", "NUM", 
                      "COMM", "WS", "DIGIT" ]

    RULE_assignment = 0
    RULE_add = 1
    RULE_add_term = 2
    RULE_term = 3
    RULE_poly_sum = 4
    RULE_poly_term = 5
    RULE_poly_mono = 6
    RULE_poly_factor = 7
    RULE_trig_sum = 8
    RULE_trig_term = 9
    RULE_trig_mono = 10
    RULE_trig_factor = 11
    RULE_exp_sum = 12
    RULE_exp_term = 13
    RULE_exp_mono = 14
    RULE_exp_factor = 15
    RULE_symvars = 16
    RULE_idd = 17
    RULE_gm = 18
    RULE_list = 19
    RULE_sub = 20
    RULE_par = 21

    ruleNames =  [ "assignment", "add", "add_term", "term", "poly_sum", 
                   "poly_term", "poly_mono", "poly_factor", "trig_sum", 
                   "trig_term", "trig_mono", "trig_factor", "exp_sum", "exp_term", 
                   "exp_mono", "exp_factor", "symvars", "idd", "gm", "list", 
                   "sub", "par" ]

    EOF = Token.EOF
    T__0=1
    T__1=2
    T__2=3
    T__3=4
    T__4=5
    T__5=6
    T__6=7
    T__7=8
    T__8=9
    T__9=10
    T__10=11
    T__11=12
    T__12=13
    T__13=14
    IDV=15
    NUM=16
    COMM=17
    WS=18
    DIGIT=19

    def __init__(self, input:TokenStream, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.10.1")
        self._interp = ParserATNSimulator(self, self.atn, self.decisionsToDFA, self.sharedContextCache)
        self._predicates = None




    class AssignmentContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def symvars(self):
            return self.getTypedRuleContext(ASGMTParser.SymvarsContext,0)


        def add(self):
            return self.getTypedRuleContext(ASGMTParser.AddContext,0)


        def poly_sum(self):
            return self.getTypedRuleContext(ASGMTParser.Poly_sumContext,0)


        def trig_sum(self):
            return self.getTypedRuleContext(ASGMTParser.Trig_sumContext,0)


        def exp_sum(self):
            return self.getTypedRuleContext(ASGMTParser.Exp_sumContext,0)


        def getRuleIndex(self):
            return ASGMTParser.RULE_assignment

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterAssignment" ):
                listener.enterAssignment(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitAssignment" ):
                listener.exitAssignment(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitAssignment" ):
                return visitor.visitAssignment(self)
            else:
                return visitor.visitChildren(self)




    def assignment(self):

        localctx = ASGMTParser.AssignmentContext(self, self._ctx, self.state)
        self.enterRule(localctx, 0, self.RULE_assignment)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 44
            self.symvars()
            self.state = 45
            self.match(ASGMTParser.T__0)
            self.state = 50
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,0,self._ctx)
            if la_ == 1:
                self.state = 46
                self.add()
                pass

            elif la_ == 2:
                self.state = 47
                self.poly_sum()
                pass

            elif la_ == 3:
                self.state = 48
                self.trig_sum()
                pass

            elif la_ == 4:
                self.state = 49
                self.exp_sum()
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class AddContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def add_term(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(ASGMTParser.Add_termContext)
            else:
                return self.getTypedRuleContext(ASGMTParser.Add_termContext,i)


        def getRuleIndex(self):
            return ASGMTParser.RULE_add

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterAdd" ):
                listener.enterAdd(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitAdd" ):
                listener.exitAdd(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitAdd" ):
                return visitor.visitAdd(self)
            else:
                return visitor.visitChildren(self)




    def add(self):

        localctx = ASGMTParser.AddContext(self, self._ctx, self.state)
        self.enterRule(localctx, 2, self.RULE_add)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 52
            self.add_term()
            self.state = 59
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,2,self._ctx)
            while _alt!=1 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1+1:
                    self.state = 54
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if _la==ASGMTParser.T__1:
                        self.state = 53
                        self.match(ASGMTParser.T__1)


                    self.state = 56
                    self.add_term() 
                self.state = 61
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,2,self._ctx)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Add_termContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def term(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(ASGMTParser.TermContext)
            else:
                return self.getTypedRuleContext(ASGMTParser.TermContext,i)


        def getRuleIndex(self):
            return ASGMTParser.RULE_add_term

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterAdd_term" ):
                listener.enterAdd_term(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitAdd_term" ):
                listener.exitAdd_term(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitAdd_term" ):
                return visitor.visitAdd_term(self)
            else:
                return visitor.visitChildren(self)




    def add_term(self):

        localctx = ASGMTParser.Add_termContext(self, self._ctx, self.state)
        self.enterRule(localctx, 4, self.RULE_add_term)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 65
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,3,self._ctx)
            if la_ == 1:
                self.state = 62
                self.term()
                self.state = 63
                self.match(ASGMTParser.T__2)


            self.state = 67
            self.term()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class TermContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def NUM(self):
            return self.getToken(ASGMTParser.NUM, 0)

        def par(self):
            return self.getTypedRuleContext(ASGMTParser.ParContext,0)


        def sub(self):
            return self.getTypedRuleContext(ASGMTParser.SubContext,0)


        def symvars(self):
            return self.getTypedRuleContext(ASGMTParser.SymvarsContext,0)


        def gm(self):
            return self.getTypedRuleContext(ASGMTParser.GmContext,0)


        def is_var(self, data):
            if not self.NUM() is None:
                return False
            if not self.par() is None:
                return False
            elif not self.symvars() is None:
                if not self.symvars().IDV() is None:
                    return True
                elif not self.symvars().idd() is None:
                    if self.symvars().idd().is_data(data):
                        return False
                    else:
                        return True
            elif not self.gm() is None:
                return True

        def is_const(self, data):
            return not self.is_var(data)

        def getValue(self, data, params_dict):
            if self.is_const(data):
                if not self.NUM() is None:
                    return float(self.NUM().getText())
                elif not self.symvars() is None:
                    return self.symvars().idd().getValue(data)
                elif not self.par() is None:
                    return self.par().getValue(params_dict)
            else:
                raise("Calling getValue for a variable")

        def getRuleIndex(self):
            return ASGMTParser.RULE_term

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterTerm" ):
                listener.enterTerm(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitTerm" ):
                listener.exitTerm(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitTerm" ):
                return visitor.visitTerm(self)
            else:
                return visitor.visitChildren(self)




    def term(self):

        localctx = ASGMTParser.TermContext(self, self._ctx, self.state)
        self.enterRule(localctx, 6, self.RULE_term)
        self._la = 0 # Token type
        try:
            self.state = 82
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,7,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 69
                self.match(ASGMTParser.NUM)
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 71
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==ASGMTParser.T__12:
                    self.state = 70
                    self.sub()


                self.state = 73
                self.par()
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 75
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==ASGMTParser.T__12:
                    self.state = 74
                    self.sub()


                self.state = 77
                self.symvars()
                pass

            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 79
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==ASGMTParser.T__12:
                    self.state = 78
                    self.sub()


                self.state = 81
                self.gm()
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Poly_sumContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def poly_term(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(ASGMTParser.Poly_termContext)
            else:
                return self.getTypedRuleContext(ASGMTParser.Poly_termContext,i)


        def getRuleIndex(self):
            return ASGMTParser.RULE_poly_sum

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterPoly_sum" ):
                listener.enterPoly_sum(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitPoly_sum" ):
                listener.exitPoly_sum(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitPoly_sum" ):
                return visitor.visitPoly_sum(self)
            else:
                return visitor.visitChildren(self)




    def poly_sum(self):

        localctx = ASGMTParser.Poly_sumContext(self, self._ctx, self.state)
        self.enterRule(localctx, 8, self.RULE_poly_sum)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 84
            self.poly_term()
            self.state = 91
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,9,self._ctx)
            while _alt!=1 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1+1:
                    self.state = 86
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if _la==ASGMTParser.T__1:
                        self.state = 85
                        self.match(ASGMTParser.T__1)


                    self.state = 88
                    self.poly_term() 
                self.state = 93
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,9,self._ctx)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Poly_termContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def poly_mono(self):
            return self.getTypedRuleContext(ASGMTParser.Poly_monoContext,0)


        def NUM(self):
            return self.getToken(ASGMTParser.NUM, 0)

        def par(self):
            return self.getTypedRuleContext(ASGMTParser.ParContext,0)


        def idd(self):
            return self.getTypedRuleContext(ASGMTParser.IddContext,0)


        def sub(self):
            return self.getTypedRuleContext(ASGMTParser.SubContext,0)


        def getCoeff(self, data, params):
            sign = -1.0 if self.sub() is not None else 1.0
            if self.NUM() is not None:
                return torch.tensor(sign * float(self.NUM().getText()))
            elif self.par() is not None:
                return torch.tensor(sign) * self.par().getValue(params)
            elif self.idd() is not None:
                return torch.tensor(sign) * self.idd().getValue(data)
            else:
                return torch.tensor(sign)

        def getRuleIndex(self):
            return ASGMTParser.RULE_poly_term

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterPoly_term" ):
                listener.enterPoly_term(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitPoly_term" ):
                listener.exitPoly_term(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitPoly_term" ):
                return visitor.visitPoly_term(self)
            else:
                return visitor.visitChildren(self)




    def poly_term(self):

        localctx = ASGMTParser.Poly_termContext(self, self._ctx, self.state)
        self.enterRule(localctx, 10, self.RULE_poly_term)
        self._la = 0 # Token type
        try:
            self.state = 116
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,15,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 95
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==ASGMTParser.T__12:
                    self.state = 94
                    self.sub()


                self.state = 100
                self._errHandler.sync(self)
                token = self._input.LA(1)
                if token in [ASGMTParser.NUM]:
                    self.state = 97
                    self.match(ASGMTParser.NUM)
                    pass
                elif token in [ASGMTParser.T__13]:
                    self.state = 98
                    self.par()
                    pass
                elif token in [ASGMTParser.IDV]:
                    self.state = 99
                    self.idd()
                    pass
                else:
                    raise NoViableAltException(self)

                self.state = 102
                self.match(ASGMTParser.T__2)
                self.state = 103
                self.poly_mono()
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 105
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==ASGMTParser.T__12:
                    self.state = 104
                    self.sub()


                self.state = 107
                self.poly_mono()
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 109
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==ASGMTParser.T__12:
                    self.state = 108
                    self.sub()


                self.state = 114
                self._errHandler.sync(self)
                token = self._input.LA(1)
                if token in [ASGMTParser.NUM]:
                    self.state = 111
                    self.match(ASGMTParser.NUM)
                    pass
                elif token in [ASGMTParser.T__13]:
                    self.state = 112
                    self.par()
                    pass
                elif token in [ASGMTParser.IDV]:
                    self.state = 113
                    self.idd()
                    pass
                else:
                    raise NoViableAltException(self)

                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Poly_monoContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def poly_factor(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(ASGMTParser.Poly_factorContext)
            else:
                return self.getTypedRuleContext(ASGMTParser.Poly_factorContext,i)


        def getRuleIndex(self):
            return ASGMTParser.RULE_poly_mono

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterPoly_mono" ):
                listener.enterPoly_mono(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitPoly_mono" ):
                listener.exitPoly_mono(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitPoly_mono" ):
                return visitor.visitPoly_mono(self)
            else:
                return visitor.visitChildren(self)




    def poly_mono(self):

        localctx = ASGMTParser.Poly_monoContext(self, self._ctx, self.state)
        self.enterRule(localctx, 12, self.RULE_poly_mono)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 118
            self.poly_factor()
            self.state = 123
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==ASGMTParser.T__2:
                self.state = 119
                self.match(ASGMTParser.T__2)
                self.state = 120
                self.poly_factor()
                self.state = 125
                self._errHandler.sync(self)
                _la = self._input.LA(1)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Poly_factorContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def symvars(self):
            return self.getTypedRuleContext(ASGMTParser.SymvarsContext,0)


        def gm(self):
            return self.getTypedRuleContext(ASGMTParser.GmContext,0)


        def NUM(self):
            return self.getToken(ASGMTParser.NUM, 0)

        def is_gm(self):
            return self.gm() is not None

        def is_var(self, data):
            if self.symvars() is not None:
                if self.symvars().idd() is not None:
                    return not self.symvars().idd().is_data(data)
                return True
            return self.is_gm()

        def getVar(self, data):
            if self.symvars() is not None:
                return self.symvars().getVar(data)
            return None

        def getExp(self):
            if self.NUM() is not None:
                return int(float(self.NUM().getText()))
            return 1

        def getRuleIndex(self):
            return ASGMTParser.RULE_poly_factor

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterPoly_factor" ):
                listener.enterPoly_factor(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitPoly_factor" ):
                listener.exitPoly_factor(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitPoly_factor" ):
                return visitor.visitPoly_factor(self)
            else:
                return visitor.visitChildren(self)




    def poly_factor(self):

        localctx = ASGMTParser.Poly_factorContext(self, self._ctx, self.state)
        self.enterRule(localctx, 14, self.RULE_poly_factor)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 128
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [ASGMTParser.IDV]:
                self.state = 126
                self.symvars()
                pass
            elif token in [ASGMTParser.T__10]:
                self.state = 127
                self.gm()
                pass
            else:
                raise NoViableAltException(self)

            self.state = 132
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==ASGMTParser.T__3:
                self.state = 130
                self.match(ASGMTParser.T__3)
                self.state = 131
                self.match(ASGMTParser.NUM)


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Trig_sumContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def trig_term(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(ASGMTParser.Trig_termContext)
            else:
                return self.getTypedRuleContext(ASGMTParser.Trig_termContext,i)


        def getRuleIndex(self):
            return ASGMTParser.RULE_trig_sum

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterTrig_sum" ):
                listener.enterTrig_sum(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitTrig_sum" ):
                listener.exitTrig_sum(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitTrig_sum" ):
                return visitor.visitTrig_sum(self)
            else:
                return visitor.visitChildren(self)




    def trig_sum(self):

        localctx = ASGMTParser.Trig_sumContext(self, self._ctx, self.state)
        self.enterRule(localctx, 16, self.RULE_trig_sum)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 134
            self.trig_term()
            self.state = 141
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,20,self._ctx)
            while _alt!=1 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1+1:
                    self.state = 136
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if _la==ASGMTParser.T__1:
                        self.state = 135
                        self.match(ASGMTParser.T__1)


                    self.state = 138
                    self.trig_term() 
                self.state = 143
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,20,self._ctx)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Trig_termContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def trig_mono(self):
            return self.getTypedRuleContext(ASGMTParser.Trig_monoContext,0)


        def NUM(self):
            return self.getToken(ASGMTParser.NUM, 0)

        def par(self):
            return self.getTypedRuleContext(ASGMTParser.ParContext,0)


        def idd(self):
            return self.getTypedRuleContext(ASGMTParser.IddContext,0)


        def sub(self):
            return self.getTypedRuleContext(ASGMTParser.SubContext,0)


        def getCoeff(self, data, params):
            sign = -1.0 if self.sub() is not None else 1.0
            if self.NUM() is not None:
                return torch.tensor(sign * float(self.NUM().getText()))
            elif self.par() is not None:
                return torch.tensor(sign) * self.par().getValue(params)
            elif self.idd() is not None:
                return torch.tensor(sign) * self.idd().getValue(data)
            else:
                return torch.tensor(sign)

        def getRuleIndex(self):
            return ASGMTParser.RULE_trig_term

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterTrig_term" ):
                listener.enterTrig_term(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitTrig_term" ):
                listener.exitTrig_term(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitTrig_term" ):
                return visitor.visitTrig_term(self)
            else:
                return visitor.visitChildren(self)




    def trig_term(self):

        localctx = ASGMTParser.Trig_termContext(self, self._ctx, self.state)
        self.enterRule(localctx, 18, self.RULE_trig_term)
        self._la = 0 # Token type
        try:
            self.state = 166
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,26,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 145
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==ASGMTParser.T__12:
                    self.state = 144
                    self.sub()


                self.state = 150
                self._errHandler.sync(self)
                token = self._input.LA(1)
                if token in [ASGMTParser.NUM]:
                    self.state = 147
                    self.match(ASGMTParser.NUM)
                    pass
                elif token in [ASGMTParser.T__13]:
                    self.state = 148
                    self.par()
                    pass
                elif token in [ASGMTParser.IDV]:
                    self.state = 149
                    self.idd()
                    pass
                else:
                    raise NoViableAltException(self)

                self.state = 152
                self.match(ASGMTParser.T__2)
                self.state = 153
                self.trig_mono()
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 155
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==ASGMTParser.T__12:
                    self.state = 154
                    self.sub()


                self.state = 157
                self.trig_mono()
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 159
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==ASGMTParser.T__12:
                    self.state = 158
                    self.sub()


                self.state = 164
                self._errHandler.sync(self)
                token = self._input.LA(1)
                if token in [ASGMTParser.NUM]:
                    self.state = 161
                    self.match(ASGMTParser.NUM)
                    pass
                elif token in [ASGMTParser.T__13]:
                    self.state = 162
                    self.par()
                    pass
                elif token in [ASGMTParser.IDV]:
                    self.state = 163
                    self.idd()
                    pass
                else:
                    raise NoViableAltException(self)

                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Trig_monoContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def trig_factor(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(ASGMTParser.Trig_factorContext)
            else:
                return self.getTypedRuleContext(ASGMTParser.Trig_factorContext,i)


        def getRuleIndex(self):
            return ASGMTParser.RULE_trig_mono

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterTrig_mono" ):
                listener.enterTrig_mono(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitTrig_mono" ):
                listener.exitTrig_mono(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitTrig_mono" ):
                return visitor.visitTrig_mono(self)
            else:
                return visitor.visitChildren(self)




    def trig_mono(self):

        localctx = ASGMTParser.Trig_monoContext(self, self._ctx, self.state)
        self.enterRule(localctx, 20, self.RULE_trig_mono)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 168
            self.trig_factor()
            self.state = 173
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==ASGMTParser.T__2:
                self.state = 169
                self.match(ASGMTParser.T__2)
                self.state = 170
                self.trig_factor()
                self.state = 175
                self._errHandler.sync(self)
                _la = self._input.LA(1)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Trig_factorContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def symvars(self):
            return self.getTypedRuleContext(ASGMTParser.SymvarsContext,0)


        def gm(self):
            return self.getTypedRuleContext(ASGMTParser.GmContext,0)


        def NUM(self):
            return self.getToken(ASGMTParser.NUM, 0)

        def is_var_factor(self):
            return not self.getText().startswith('cos(') and not self.getText().startswith('sin(')

        def is_cos(self):
            return self.getText().startswith('cos(')

        def is_sin(self):
            return self.getText().startswith('sin(')

        def is_gm(self):
            return self.gm() is not None

        def is_var(self, data):
            if self.symvars() is not None:
                if self.symvars().idd() is not None:
                    return not self.symvars().idd().is_data(data)
                return True
            return self.is_gm()

        def getVar(self, data):
            if self.symvars() is not None:
                return self.symvars().getVar(data)
            return None

        def getExp(self):
            if self.NUM() is not None:
                return int(float(self.NUM().getText()))
            return 1

        def getRuleIndex(self):
            return ASGMTParser.RULE_trig_factor

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterTrig_factor" ):
                listener.enterTrig_factor(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitTrig_factor" ):
                listener.exitTrig_factor(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitTrig_factor" ):
                return visitor.visitTrig_factor(self)
            else:
                return visitor.visitChildren(self)




    def trig_factor(self):

        localctx = ASGMTParser.Trig_factorContext(self, self._ctx, self.state)
        self.enterRule(localctx, 22, self.RULE_trig_factor)
        self._la = 0 # Token type
        try:
            self.state = 204
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [ASGMTParser.T__10, ASGMTParser.IDV]:
                self.enterOuterAlt(localctx, 1)
                self.state = 178
                self._errHandler.sync(self)
                token = self._input.LA(1)
                if token in [ASGMTParser.IDV]:
                    self.state = 176
                    self.symvars()
                    pass
                elif token in [ASGMTParser.T__10]:
                    self.state = 177
                    self.gm()
                    pass
                else:
                    raise NoViableAltException(self)

                self.state = 182
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==ASGMTParser.T__3:
                    self.state = 180
                    self.match(ASGMTParser.T__3)
                    self.state = 181
                    self.match(ASGMTParser.NUM)


                pass
            elif token in [ASGMTParser.T__4, ASGMTParser.T__6]:
                self.enterOuterAlt(localctx, 2)
                self.state = 198
                self._errHandler.sync(self)
                token = self._input.LA(1)
                if token in [ASGMTParser.T__4]:
                    self.state = 184
                    self.match(ASGMTParser.T__4)
                    self.state = 187
                    self._errHandler.sync(self)
                    token = self._input.LA(1)
                    if token in [ASGMTParser.IDV]:
                        self.state = 185
                        self.symvars()
                        pass
                    elif token in [ASGMTParser.T__10]:
                        self.state = 186
                        self.gm()
                        pass
                    else:
                        raise NoViableAltException(self)

                    self.state = 189
                    self.match(ASGMTParser.T__5)
                    pass
                elif token in [ASGMTParser.T__6]:
                    self.state = 191
                    self.match(ASGMTParser.T__6)
                    self.state = 194
                    self._errHandler.sync(self)
                    token = self._input.LA(1)
                    if token in [ASGMTParser.IDV]:
                        self.state = 192
                        self.symvars()
                        pass
                    elif token in [ASGMTParser.T__10]:
                        self.state = 193
                        self.gm()
                        pass
                    else:
                        raise NoViableAltException(self)

                    self.state = 196
                    self.match(ASGMTParser.T__5)
                    pass
                else:
                    raise NoViableAltException(self)

                self.state = 202
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==ASGMTParser.T__3:
                    self.state = 200
                    self.match(ASGMTParser.T__3)
                    self.state = 201
                    self.match(ASGMTParser.NUM)


                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Exp_sumContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def exp_term(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(ASGMTParser.Exp_termContext)
            else:
                return self.getTypedRuleContext(ASGMTParser.Exp_termContext,i)


        def getRuleIndex(self):
            return ASGMTParser.RULE_exp_sum

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterExp_sum" ):
                listener.enterExp_sum(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitExp_sum" ):
                listener.exitExp_sum(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitExp_sum" ):
                return visitor.visitExp_sum(self)
            else:
                return visitor.visitChildren(self)




    def exp_sum(self):

        localctx = ASGMTParser.Exp_sumContext(self, self._ctx, self.state)
        self.enterRule(localctx, 24, self.RULE_exp_sum)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 206
            self.exp_term()
            self.state = 213
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,36,self._ctx)
            while _alt!=1 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1+1:
                    self.state = 208
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if _la==ASGMTParser.T__1:
                        self.state = 207
                        self.match(ASGMTParser.T__1)


                    self.state = 210
                    self.exp_term() 
                self.state = 215
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,36,self._ctx)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Exp_termContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def exp_mono(self):
            return self.getTypedRuleContext(ASGMTParser.Exp_monoContext,0)


        def NUM(self):
            return self.getToken(ASGMTParser.NUM, 0)

        def par(self):
            return self.getTypedRuleContext(ASGMTParser.ParContext,0)


        def idd(self):
            return self.getTypedRuleContext(ASGMTParser.IddContext,0)


        def sub(self):
            return self.getTypedRuleContext(ASGMTParser.SubContext,0)


        def getCoeff(self, data, params):
            sign = -1.0 if self.sub() is not None else 1.0
            if self.NUM() is not None:
                return torch.tensor(sign * float(self.NUM().getText()))
            elif self.par() is not None:
                return torch.tensor(sign) * self.par().getValue(params)
            elif self.idd() is not None:
                return torch.tensor(sign) * self.idd().getValue(data)
            else:
                return torch.tensor(sign)

        def getRuleIndex(self):
            return ASGMTParser.RULE_exp_term

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterExp_term" ):
                listener.enterExp_term(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitExp_term" ):
                listener.exitExp_term(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitExp_term" ):
                return visitor.visitExp_term(self)
            else:
                return visitor.visitChildren(self)




    def exp_term(self):

        localctx = ASGMTParser.Exp_termContext(self, self._ctx, self.state)
        self.enterRule(localctx, 26, self.RULE_exp_term)
        self._la = 0 # Token type
        try:
            self.state = 238
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,42,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 217
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==ASGMTParser.T__12:
                    self.state = 216
                    self.sub()


                self.state = 222
                self._errHandler.sync(self)
                token = self._input.LA(1)
                if token in [ASGMTParser.NUM]:
                    self.state = 219
                    self.match(ASGMTParser.NUM)
                    pass
                elif token in [ASGMTParser.T__13]:
                    self.state = 220
                    self.par()
                    pass
                elif token in [ASGMTParser.IDV]:
                    self.state = 221
                    self.idd()
                    pass
                else:
                    raise NoViableAltException(self)

                self.state = 224
                self.match(ASGMTParser.T__2)
                self.state = 225
                self.exp_mono()
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 227
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==ASGMTParser.T__12:
                    self.state = 226
                    self.sub()


                self.state = 229
                self.exp_mono()
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 231
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==ASGMTParser.T__12:
                    self.state = 230
                    self.sub()


                self.state = 236
                self._errHandler.sync(self)
                token = self._input.LA(1)
                if token in [ASGMTParser.NUM]:
                    self.state = 233
                    self.match(ASGMTParser.NUM)
                    pass
                elif token in [ASGMTParser.T__13]:
                    self.state = 234
                    self.par()
                    pass
                elif token in [ASGMTParser.IDV]:
                    self.state = 235
                    self.idd()
                    pass
                else:
                    raise NoViableAltException(self)

                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Exp_monoContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def exp_factor(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(ASGMTParser.Exp_factorContext)
            else:
                return self.getTypedRuleContext(ASGMTParser.Exp_factorContext,i)


        def getRuleIndex(self):
            return ASGMTParser.RULE_exp_mono

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterExp_mono" ):
                listener.enterExp_mono(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitExp_mono" ):
                listener.exitExp_mono(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitExp_mono" ):
                return visitor.visitExp_mono(self)
            else:
                return visitor.visitChildren(self)




    def exp_mono(self):

        localctx = ASGMTParser.Exp_monoContext(self, self._ctx, self.state)
        self.enterRule(localctx, 28, self.RULE_exp_mono)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 240
            self.exp_factor()
            self.state = 245
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==ASGMTParser.T__2:
                self.state = 241
                self.match(ASGMTParser.T__2)
                self.state = 242
                self.exp_factor()
                self.state = 247
                self._errHandler.sync(self)
                _la = self._input.LA(1)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Exp_factorContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def symvars(self):
            return self.getTypedRuleContext(ASGMTParser.SymvarsContext,0)


        def gm(self):
            return self.getTypedRuleContext(ASGMTParser.GmContext,0)


        def NUM(self):
            return self.getToken(ASGMTParser.NUM, 0)

        def is_var_factor(self):
            return not self.getText().startswith('exp(')

        def is_exp(self):
            return self.getText().startswith('exp(')

        def is_gm(self):
            return self.gm() is not None

        def is_var(self, data):
            if self.symvars() is not None:
                if self.symvars().idd() is not None:
                    return not self.symvars().idd().is_data(data)
                return True
            return self.is_gm()

        def getVar(self, data):
            if self.symvars() is not None:
                return self.symvars().getVar(data)
            return None

        def getExp(self):
            if self.NUM() is not None:
                return int(float(self.NUM().getText()))
            return 1

        def getRuleIndex(self):
            return ASGMTParser.RULE_exp_factor

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterExp_factor" ):
                listener.enterExp_factor(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitExp_factor" ):
                listener.exitExp_factor(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitExp_factor" ):
                return visitor.visitExp_factor(self)
            else:
                return visitor.visitChildren(self)




    def exp_factor(self):

        localctx = ASGMTParser.Exp_factorContext(self, self._ctx, self.state)
        self.enterRule(localctx, 30, self.RULE_exp_factor)
        self._la = 0 # Token type
        try:
            self.state = 266
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [ASGMTParser.T__10, ASGMTParser.IDV]:
                self.enterOuterAlt(localctx, 1)
                self.state = 250
                self._errHandler.sync(self)
                token = self._input.LA(1)
                if token in [ASGMTParser.IDV]:
                    self.state = 248
                    self.symvars()
                    pass
                elif token in [ASGMTParser.T__10]:
                    self.state = 249
                    self.gm()
                    pass
                else:
                    raise NoViableAltException(self)

                self.state = 254
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==ASGMTParser.T__3:
                    self.state = 252
                    self.match(ASGMTParser.T__3)
                    self.state = 253
                    self.match(ASGMTParser.NUM)


                pass
            elif token in [ASGMTParser.T__7]:
                self.enterOuterAlt(localctx, 2)
                self.state = 256
                self.match(ASGMTParser.T__7)
                self.state = 259
                self._errHandler.sync(self)
                token = self._input.LA(1)
                if token in [ASGMTParser.IDV]:
                    self.state = 257
                    self.symvars()
                    pass
                elif token in [ASGMTParser.T__10]:
                    self.state = 258
                    self.gm()
                    pass
                else:
                    raise NoViableAltException(self)

                self.state = 261
                self.match(ASGMTParser.T__5)
                self.state = 264
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==ASGMTParser.T__3:
                    self.state = 262
                    self.match(ASGMTParser.T__3)
                    self.state = 263
                    self.match(ASGMTParser.NUM)


                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class SymvarsContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def IDV(self):
            return self.getToken(ASGMTParser.IDV, 0)

        def idd(self):
            return self.getTypedRuleContext(ASGMTParser.IddContext,0)


        def getVar(self, data):
            if self.idd() is None:
                return self.getText()
            else:
                if self.idd().IDV(1) is None:
                    return self.getText()
                else:
                    data_idx = int(data[self.idd().IDV(1).getText()][0].item())
                return self.idd().IDV(0).getText()+'['+str(data_idx)+']'

        def getRuleIndex(self):
            return ASGMTParser.RULE_symvars

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterSymvars" ):
                listener.enterSymvars(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitSymvars" ):
                listener.exitSymvars(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitSymvars" ):
                return visitor.visitSymvars(self)
            else:
                return visitor.visitChildren(self)




    def symvars(self):

        localctx = ASGMTParser.SymvarsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 32, self.RULE_symvars)
        try:
            self.state = 270
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,49,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 268
                self.match(ASGMTParser.IDV)
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 269
                self.idd()
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class IddContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def IDV(self, i:int=None):
            if i is None:
                return self.getTokens(ASGMTParser.IDV)
            else:
                return self.getToken(ASGMTParser.IDV, i)

        def NUM(self):
            return self.getToken(ASGMTParser.NUM, 0)

        def is_data(self, data):
            if self.IDV(0).getText() in data.keys():
                return True
            else:
                return False

        def getValue(self, data):
            data_name = self.IDV(0).getText()
            if not self.NUM() is None:
                data_idx = int(self.NUM().getText())
            elif not self.IDV(1) is None:
                data_idx = int(data[self.IDV(1).getText()][0].item())
            return data[data_name][data_idx]

        def getRuleIndex(self):
            return ASGMTParser.RULE_idd

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterIdd" ):
                listener.enterIdd(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitIdd" ):
                listener.exitIdd(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitIdd" ):
                return visitor.visitIdd(self)
            else:
                return visitor.visitChildren(self)




    def idd(self):

        localctx = ASGMTParser.IddContext(self, self._ctx, self.state)
        self.enterRule(localctx, 34, self.RULE_idd)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 272
            self.match(ASGMTParser.IDV)
            self.state = 273
            self.match(ASGMTParser.T__8)
            self.state = 274
            _la = self._input.LA(1)
            if not(_la==ASGMTParser.IDV or _la==ASGMTParser.NUM):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
            self.state = 275
            self.match(ASGMTParser.T__9)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class GmContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def list_(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(ASGMTParser.ListContext)
            else:
                return self.getTypedRuleContext(ASGMTParser.ListContext,i)


        def getRuleIndex(self):
            return ASGMTParser.RULE_gm

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterGm" ):
                listener.enterGm(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitGm" ):
                listener.exitGm(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitGm" ):
                return visitor.visitGm(self)
            else:
                return visitor.visitChildren(self)




    def gm(self):

        localctx = ASGMTParser.GmContext(self, self._ctx, self.state)
        self.enterRule(localctx, 36, self.RULE_gm)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 277
            self.match(ASGMTParser.T__10)
            self.state = 278
            self.list_()
            self.state = 279
            self.match(ASGMTParser.T__11)
            self.state = 280
            self.list_()
            self.state = 281
            self.match(ASGMTParser.T__11)
            self.state = 282
            self.list_()
            self.state = 283
            self.match(ASGMTParser.T__5)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ListContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def NUM(self, i:int=None):
            if i is None:
                return self.getTokens(ASGMTParser.NUM)
            else:
                return self.getToken(ASGMTParser.NUM, i)

        def par(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(ASGMTParser.ParContext)
            else:
                return self.getTypedRuleContext(ASGMTParser.ParContext,i)


        def unpack(self, params_dict):
            str_list = self.getText()[1:-1].split(',')
            unpacked = torch.zeros(len(str_list))
            for i, elem in enumerate(str_list):
                if elem[0] == '_':
                    unpacked[i] = params_dict[elem[1:]]
                else:
                    unpacked[i] = float(elem)
            return unpacked

        def getRuleIndex(self):
            return ASGMTParser.RULE_list

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterList" ):
                listener.enterList(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitList" ):
                listener.exitList(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitList" ):
                return visitor.visitList(self)
            else:
                return visitor.visitChildren(self)




    def list_(self):

        localctx = ASGMTParser.ListContext(self, self._ctx, self.state)
        self.enterRule(localctx, 38, self.RULE_list)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 285
            self.match(ASGMTParser.T__8)
            self.state = 288
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [ASGMTParser.NUM]:
                self.state = 286
                self.match(ASGMTParser.NUM)
                pass
            elif token in [ASGMTParser.T__13]:
                self.state = 287
                self.par()
                pass
            else:
                raise NoViableAltException(self)

            self.state = 297
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,52,self._ctx)
            while _alt!=1 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1+1:
                    self.state = 290
                    self.match(ASGMTParser.T__11)
                    self.state = 293
                    self._errHandler.sync(self)
                    token = self._input.LA(1)
                    if token in [ASGMTParser.NUM]:
                        self.state = 291
                        self.match(ASGMTParser.NUM)
                        pass
                    elif token in [ASGMTParser.T__13]:
                        self.state = 292
                        self.par()
                        pass
                    else:
                        raise NoViableAltException(self)
             
                self.state = 299
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,52,self._ctx)

            self.state = 300
            self.match(ASGMTParser.T__9)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class SubContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return ASGMTParser.RULE_sub

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterSub" ):
                listener.enterSub(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitSub" ):
                listener.exitSub(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitSub" ):
                return visitor.visitSub(self)
            else:
                return visitor.visitChildren(self)




    def sub(self):

        localctx = ASGMTParser.SubContext(self, self._ctx, self.state)
        self.enterRule(localctx, 40, self.RULE_sub)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 302
            self.match(ASGMTParser.T__12)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ParContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def IDV(self):
            return self.getToken(ASGMTParser.IDV, 0)

        def getValue(self, params_dict):
            name = self.getText()
            name = name[1:]
            return params_dict[name]

        def getRuleIndex(self):
            return ASGMTParser.RULE_par

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterPar" ):
                listener.enterPar(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitPar" ):
                listener.exitPar(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitPar" ):
                return visitor.visitPar(self)
            else:
                return visitor.visitChildren(self)




    def par(self):

        localctx = ASGMTParser.ParContext(self, self._ctx, self.state)
        self.enterRule(localctx, 42, self.RULE_par)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 304
            self.match(ASGMTParser.T__13)
            self.state = 305
            self.match(ASGMTParser.IDV)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx





