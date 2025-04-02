# Generated from TRUNC.g4 by ANTLR 4.10.1
# encoding: utf-8
from antlr4 import *
from io import StringIO
import sys
if sys.version_info[1] > 5:
	from typing import TextIO
else:
	from typing.io import TextIO
import torch

def serializedATN():
    return [
        4,1,23,156,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,6,7,
        6,2,7,7,7,2,8,7,8,2,9,7,9,2,10,7,10,2,11,7,11,2,12,7,12,2,13,7,13,
        2,14,7,14,2,15,7,15,2,16,7,16,2,17,7,17,1,0,1,0,1,0,1,0,3,0,41,8,
        0,1,1,1,1,1,1,1,1,1,2,1,2,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,4,1,
        4,1,4,1,4,1,4,1,4,1,4,1,4,1,5,1,5,1,5,1,5,1,6,1,6,1,7,1,7,1,7,3,
        7,74,8,7,1,7,1,7,5,7,78,8,7,10,7,12,7,81,9,7,1,8,1,8,1,8,3,8,86,
        8,8,1,8,3,8,89,8,8,1,8,1,8,1,9,1,9,1,9,5,9,96,8,9,10,9,12,9,99,9,
        9,1,10,1,10,1,10,3,10,104,8,10,1,10,1,10,1,10,1,10,3,10,110,8,10,
        3,10,112,8,10,1,11,1,11,1,11,3,11,117,8,11,1,12,1,12,1,12,1,12,1,
        12,1,13,1,13,1,13,1,13,1,13,1,13,1,13,1,13,1,14,1,14,1,14,3,14,135,
        8,14,1,14,1,14,1,14,3,14,140,8,14,5,14,142,8,14,10,14,12,14,145,
        9,14,1,14,1,14,1,15,1,15,1,16,1,16,1,17,1,17,1,17,1,17,3,79,97,143,
        0,18,0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,0,4,1,0,1,
        4,1,0,7,8,1,0,10,11,1,0,18,19,156,0,40,1,0,0,0,2,42,1,0,0,0,4,46,
        1,0,0,0,6,48,1,0,0,0,8,56,1,0,0,0,10,64,1,0,0,0,12,68,1,0,0,0,14,
        70,1,0,0,0,16,88,1,0,0,0,18,92,1,0,0,0,20,103,1,0,0,0,22,116,1,0,
        0,0,24,118,1,0,0,0,26,123,1,0,0,0,28,131,1,0,0,0,30,148,1,0,0,0,
        32,150,1,0,0,0,34,152,1,0,0,0,36,41,3,2,1,0,37,41,3,10,5,0,38,41,
        3,6,3,0,39,41,3,8,4,0,40,36,1,0,0,0,40,37,1,0,0,0,40,38,1,0,0,0,
        40,39,1,0,0,0,41,1,1,0,0,0,42,43,3,14,7,0,43,44,3,4,2,0,44,45,3,
        18,9,0,45,3,1,0,0,0,46,47,7,0,0,0,47,5,1,0,0,0,48,49,5,18,0,0,49,
        50,3,4,2,0,50,51,3,18,9,0,51,52,5,5,0,0,52,53,5,18,0,0,53,54,3,4,
        2,0,54,55,3,18,9,0,55,7,1,0,0,0,56,57,5,18,0,0,57,58,3,4,2,0,58,
        59,3,18,9,0,59,60,5,6,0,0,60,61,5,18,0,0,61,62,3,4,2,0,62,63,3,18,
        9,0,63,9,1,0,0,0,64,65,3,22,11,0,65,66,3,12,6,0,66,67,3,18,9,0,67,
        11,1,0,0,0,68,69,7,1,0,0,69,13,1,0,0,0,70,79,3,16,8,0,71,74,3,30,
        15,0,72,74,3,32,16,0,73,71,1,0,0,0,73,72,1,0,0,0,74,75,1,0,0,0,75,
        76,3,16,8,0,76,78,1,0,0,0,77,73,1,0,0,0,78,81,1,0,0,0,79,80,1,0,
        0,0,79,77,1,0,0,0,80,15,1,0,0,0,81,79,1,0,0,0,82,86,5,19,0,0,83,
        86,3,34,17,0,84,86,3,24,12,0,85,82,1,0,0,0,85,83,1,0,0,0,85,84,1,
        0,0,0,86,87,1,0,0,0,87,89,5,9,0,0,88,85,1,0,0,0,88,89,1,0,0,0,89,
        90,1,0,0,0,90,91,3,22,11,0,91,17,1,0,0,0,92,97,3,20,10,0,93,94,7,
        2,0,0,94,96,3,20,10,0,95,93,1,0,0,0,96,99,1,0,0,0,97,98,1,0,0,0,
        97,95,1,0,0,0,98,19,1,0,0,0,99,97,1,0,0,0,100,104,5,19,0,0,101,104,
        3,34,17,0,102,104,3,24,12,0,103,100,1,0,0,0,103,101,1,0,0,0,103,
        102,1,0,0,0,104,111,1,0,0,0,105,109,5,9,0,0,106,110,5,19,0,0,107,
        110,3,24,12,0,108,110,3,34,17,0,109,106,1,0,0,0,109,107,1,0,0,0,
        109,108,1,0,0,0,110,112,1,0,0,0,111,105,1,0,0,0,111,112,1,0,0,0,
        112,21,1,0,0,0,113,117,5,18,0,0,114,117,3,24,12,0,115,117,3,26,13,
        0,116,113,1,0,0,0,116,114,1,0,0,0,116,115,1,0,0,0,117,23,1,0,0,0,
        118,119,5,18,0,0,119,120,5,12,0,0,120,121,7,3,0,0,121,122,5,13,0,
        0,122,25,1,0,0,0,123,124,5,14,0,0,124,125,3,28,14,0,125,126,5,15,
        0,0,126,127,3,28,14,0,127,128,5,15,0,0,128,129,3,28,14,0,129,130,
        5,16,0,0,130,27,1,0,0,0,131,134,5,12,0,0,132,135,5,19,0,0,133,135,
        3,34,17,0,134,132,1,0,0,0,134,133,1,0,0,0,135,143,1,0,0,0,136,139,
        5,15,0,0,137,140,5,19,0,0,138,140,3,34,17,0,139,137,1,0,0,0,139,
        138,1,0,0,0,140,142,1,0,0,0,141,136,1,0,0,0,142,145,1,0,0,0,143,
        144,1,0,0,0,143,141,1,0,0,0,144,146,1,0,0,0,145,143,1,0,0,0,146,
        147,5,13,0,0,147,29,1,0,0,0,148,149,5,10,0,0,149,31,1,0,0,0,150,
        151,5,11,0,0,151,33,1,0,0,0,152,153,5,17,0,0,153,154,5,18,0,0,154,
        35,1,0,0,0,13,40,73,79,85,88,97,103,109,111,116,134,139,143
    ]

class TRUNCParser ( Parser ):

    grammarFileName = "TRUNC.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    sharedContextCache = PredictionContextCache()

    literalNames = [ "<INVALID>", "'<='", "'<'", "'>'", "'>='", "'and'", 
                     "'or'", "'=='", "'!='", "'*'", "'+'", "'-'", "'['", 
                     "']'", "'gm('", "','", "')'", "'_'" ]

    symbolicNames = [ "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "IDV", "NUM", "COMM", "WS", 
                      "ALPHA", "DIGIT" ]

    RULE_trunc = 0
    RULE_ineq = 1
    RULE_inop = 2
    RULE_and_trunc = 3
    RULE_or_trunc = 4
    RULE_eq = 5
    RULE_eqop = 6
    RULE_lexpr = 7
    RULE_monom = 8
    RULE_const = 9
    RULE_const_term = 10
    RULE_var = 11
    RULE_idd = 12
    RULE_gm = 13
    RULE_list = 14
    RULE_sum = 15
    RULE_sub = 16
    RULE_par = 17

    ruleNames =  [ "trunc", "ineq", "inop", "and_trunc", "or_trunc", "eq", 
                   "eqop", "lexpr", "monom", "const", "const_term", "var", 
                   "idd", "gm", "list", "sum", "sub", "par" ]

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
    T__14=15
    T__15=16
    T__16=17
    IDV=18
    NUM=19
    COMM=20
    WS=21
    ALPHA=22
    DIGIT=23

    def __init__(self, input:TokenStream, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.10.1")
        self._interp = ParserATNSimulator(self, self.atn, self.decisionsToDFA, self.sharedContextCache)
        self._predicates = None




    class TruncContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ineq(self):
            return self.getTypedRuleContext(TRUNCParser.IneqContext,0)


        def eq(self):
            return self.getTypedRuleContext(TRUNCParser.EqContext,0)


        def and_trunc(self):
            return self.getTypedRuleContext(TRUNCParser.And_truncContext,0)


        def or_trunc(self):
            return self.getTypedRuleContext(TRUNCParser.Or_truncContext,0)


        def getRuleIndex(self):
            return TRUNCParser.RULE_trunc

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterTrunc" ):
                listener.enterTrunc(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitTrunc" ):
                listener.exitTrunc(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitTrunc" ):
                return visitor.visitTrunc(self)
            else:
                return visitor.visitChildren(self)




    def trunc(self):

        localctx = TRUNCParser.TruncContext(self, self._ctx, self.state)
        self.enterRule(localctx, 0, self.RULE_trunc)
        try:
            self.state = 40
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,0,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 36
                self.ineq()
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 37
                self.eq()
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 38
                self.and_trunc()
                pass

            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 39
                self.or_trunc()
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class IneqContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def lexpr(self):
            return self.getTypedRuleContext(TRUNCParser.LexprContext,0)


        def inop(self):
            return self.getTypedRuleContext(TRUNCParser.InopContext,0)


        def const(self):
            return self.getTypedRuleContext(TRUNCParser.ConstContext,0)


        def getRuleIndex(self):
            return TRUNCParser.RULE_ineq

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterIneq" ):
                listener.enterIneq(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitIneq" ):
                listener.exitIneq(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitIneq" ):
                return visitor.visitIneq(self)
            else:
                return visitor.visitChildren(self)




    def ineq(self):

        localctx = TRUNCParser.IneqContext(self, self._ctx, self.state)
        self.enterRule(localctx, 2, self.RULE_ineq)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 42
            self.lexpr()
            self.state = 43
            self.inop()
            self.state = 44
            self.const()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class InopContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return TRUNCParser.RULE_inop

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterInop" ):
                listener.enterInop(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitInop" ):
                listener.exitInop(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitInop" ):
                return visitor.visitInop(self)
            else:
                return visitor.visitChildren(self)




    def inop(self):

        localctx = TRUNCParser.InopContext(self, self._ctx, self.state)
        self.enterRule(localctx, 4, self.RULE_inop)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 46
            _la = self._input.LA(1)
            if not((((_la) & ~0x3f) == 0 and ((1 << _la) & ((1 << TRUNCParser.T__0) | (1 << TRUNCParser.T__1) | (1 << TRUNCParser.T__2) | (1 << TRUNCParser.T__3))) != 0)):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class And_truncContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def IDV(self, i:int=None):
            if i is None:
                return self.getTokens(TRUNCParser.IDV)
            else:
                return self.getToken(TRUNCParser.IDV, i)

        def inop(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(TRUNCParser.InopContext)
            else:
                return self.getTypedRuleContext(TRUNCParser.InopContext,i)


        def const(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(TRUNCParser.ConstContext)
            else:
                return self.getTypedRuleContext(TRUNCParser.ConstContext,i)


        def getRuleIndex(self):
            return TRUNCParser.RULE_and_trunc

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterAnd_trunc" ):
                listener.enterAnd_trunc(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitAnd_trunc" ):
                listener.exitAnd_trunc(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitAnd_trunc" ):
                return visitor.visitAnd_trunc(self)
            else:
                return visitor.visitChildren(self)




    def and_trunc(self):

        localctx = TRUNCParser.And_truncContext(self, self._ctx, self.state)
        self.enterRule(localctx, 6, self.RULE_and_trunc)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 48
            self.match(TRUNCParser.IDV)
            self.state = 49
            self.inop()
            self.state = 50
            self.const()
            self.state = 51
            self.match(TRUNCParser.T__4)
            self.state = 52
            self.match(TRUNCParser.IDV)
            self.state = 53
            self.inop()
            self.state = 54
            self.const()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Or_truncContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def IDV(self, i:int=None):
            if i is None:
                return self.getTokens(TRUNCParser.IDV)
            else:
                return self.getToken(TRUNCParser.IDV, i)

        def inop(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(TRUNCParser.InopContext)
            else:
                return self.getTypedRuleContext(TRUNCParser.InopContext,i)


        def const(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(TRUNCParser.ConstContext)
            else:
                return self.getTypedRuleContext(TRUNCParser.ConstContext,i)


        def getRuleIndex(self):
            return TRUNCParser.RULE_or_trunc

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterOr_trunc" ):
                listener.enterOr_trunc(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitOr_trunc" ):
                listener.exitOr_trunc(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitOr_trunc" ):
                return visitor.visitOr_trunc(self)
            else:
                return visitor.visitChildren(self)




    def or_trunc(self):

        localctx = TRUNCParser.Or_truncContext(self, self._ctx, self.state)
        self.enterRule(localctx, 8, self.RULE_or_trunc)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 56
            self.match(TRUNCParser.IDV)
            self.state = 57
            self.inop()
            self.state = 58
            self.const()
            self.state = 59
            self.match(TRUNCParser.T__5)
            self.state = 60
            self.match(TRUNCParser.IDV)
            self.state = 61
            self.inop()
            self.state = 62
            self.const()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class EqContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def var(self):
            return self.getTypedRuleContext(TRUNCParser.VarContext,0)


        def eqop(self):
            return self.getTypedRuleContext(TRUNCParser.EqopContext,0)


        def const(self):
            return self.getTypedRuleContext(TRUNCParser.ConstContext,0)


        def getRuleIndex(self):
            return TRUNCParser.RULE_eq

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterEq" ):
                listener.enterEq(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitEq" ):
                listener.exitEq(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitEq" ):
                return visitor.visitEq(self)
            else:
                return visitor.visitChildren(self)




    def eq(self):

        localctx = TRUNCParser.EqContext(self, self._ctx, self.state)
        self.enterRule(localctx, 10, self.RULE_eq)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 64
            self.var()
            self.state = 65
            self.eqop()
            self.state = 66
            self.const()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class EqopContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return TRUNCParser.RULE_eqop

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterEqop" ):
                listener.enterEqop(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitEqop" ):
                listener.exitEqop(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitEqop" ):
                return visitor.visitEqop(self)
            else:
                return visitor.visitChildren(self)




    def eqop(self):

        localctx = TRUNCParser.EqopContext(self, self._ctx, self.state)
        self.enterRule(localctx, 12, self.RULE_eqop)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 68
            _la = self._input.LA(1)
            if not(_la==TRUNCParser.T__6 or _la==TRUNCParser.T__7):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class LexprContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def monom(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(TRUNCParser.MonomContext)
            else:
                return self.getTypedRuleContext(TRUNCParser.MonomContext,i)


        def sum_(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(TRUNCParser.SumContext)
            else:
                return self.getTypedRuleContext(TRUNCParser.SumContext,i)


        def sub(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(TRUNCParser.SubContext)
            else:
                return self.getTypedRuleContext(TRUNCParser.SubContext,i)


        def getRuleIndex(self):
            return TRUNCParser.RULE_lexpr

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterLexpr" ):
                listener.enterLexpr(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitLexpr" ):
                listener.exitLexpr(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitLexpr" ):
                return visitor.visitLexpr(self)
            else:
                return visitor.visitChildren(self)




    def lexpr(self):

        localctx = TRUNCParser.LexprContext(self, self._ctx, self.state)
        self.enterRule(localctx, 14, self.RULE_lexpr)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 70
            self.monom()
            self.state = 79
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,2,self._ctx)
            while _alt!=1 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1+1:
                    self.state = 73
                    self._errHandler.sync(self)
                    token = self._input.LA(1)
                    if token in [TRUNCParser.T__9]:
                        self.state = 71
                        self.sum_()
                        pass
                    elif token in [TRUNCParser.T__10]:
                        self.state = 72
                        self.sub()
                        pass
                    else:
                        raise NoViableAltException(self)

                    self.state = 75
                    self.monom() 
                self.state = 81
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,2,self._ctx)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class MonomContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def var(self):
            return self.getTypedRuleContext(TRUNCParser.VarContext,0)


        def NUM(self):
            return self.getToken(TRUNCParser.NUM, 0)

        def par(self):
            return self.getTypedRuleContext(TRUNCParser.ParContext,0)


        def idd(self):
            return self.getTypedRuleContext(TRUNCParser.IddContext,0)


        def getRuleIndex(self):
            return TRUNCParser.RULE_monom

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterMonom" ):
                listener.enterMonom(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitMonom" ):
                listener.exitMonom(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitMonom" ):
                return visitor.visitMonom(self)
            else:
                return visitor.visitChildren(self)




    def monom(self):

        localctx = TRUNCParser.MonomContext(self, self._ctx, self.state)
        self.enterRule(localctx, 16, self.RULE_monom)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 88
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,4,self._ctx)
            if la_ == 1:
                self.state = 85
                self._errHandler.sync(self)
                token = self._input.LA(1)
                if token in [TRUNCParser.NUM]:
                    self.state = 82
                    self.match(TRUNCParser.NUM)
                    pass
                elif token in [TRUNCParser.T__16]:
                    self.state = 83
                    self.par()
                    pass
                elif token in [TRUNCParser.IDV]:
                    self.state = 84
                    self.idd()
                    pass
                else:
                    raise NoViableAltException(self)

                self.state = 87
                self.match(TRUNCParser.T__8)


            self.state = 90
            self.var()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ConstContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def const_term(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(TRUNCParser.Const_termContext)
            else:
                return self.getTypedRuleContext(TRUNCParser.Const_termContext,i)


        def getRuleIndex(self):
            return TRUNCParser.RULE_const

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterConst" ):
                listener.enterConst(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitConst" ):
                listener.exitConst(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitConst" ):
                return visitor.visitConst(self)
            else:
                return visitor.visitChildren(self)




    def const(self):

        localctx = TRUNCParser.ConstContext(self, self._ctx, self.state)
        self.enterRule(localctx, 18, self.RULE_const)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 92
            self.const_term()
            self.state = 97
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,5,self._ctx)
            while _alt!=1 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1+1:
                    self.state = 93
                    _la = self._input.LA(1)
                    if not(_la==TRUNCParser.T__9 or _la==TRUNCParser.T__10):
                        self._errHandler.recoverInline(self)
                    else:
                        self._errHandler.reportMatch(self)
                        self.consume()
                    self.state = 94
                    self.const_term() 
                self.state = 99
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,5,self._ctx)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Const_termContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def NUM(self, i:int=None):
            if i is None:
                return self.getTokens(TRUNCParser.NUM)
            else:
                return self.getToken(TRUNCParser.NUM, i)

        def par(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(TRUNCParser.ParContext)
            else:
                return self.getTypedRuleContext(TRUNCParser.ParContext,i)


        def idd(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(TRUNCParser.IddContext)
            else:
                return self.getTypedRuleContext(TRUNCParser.IddContext,i)


        def getRuleIndex(self):
            return TRUNCParser.RULE_const_term

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterConst_term" ):
                listener.enterConst_term(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitConst_term" ):
                listener.exitConst_term(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitConst_term" ):
                return visitor.visitConst_term(self)
            else:
                return visitor.visitChildren(self)




    def const_term(self):

        localctx = TRUNCParser.Const_termContext(self, self._ctx, self.state)
        self.enterRule(localctx, 20, self.RULE_const_term)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 103
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [TRUNCParser.NUM]:
                self.state = 100
                self.match(TRUNCParser.NUM)
                pass
            elif token in [TRUNCParser.T__16]:
                self.state = 101
                self.par()
                pass
            elif token in [TRUNCParser.IDV]:
                self.state = 102
                self.idd()
                pass
            else:
                raise NoViableAltException(self)

            self.state = 111
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==TRUNCParser.T__8:
                self.state = 105
                self.match(TRUNCParser.T__8)
                self.state = 109
                self._errHandler.sync(self)
                token = self._input.LA(1)
                if token in [TRUNCParser.NUM]:
                    self.state = 106
                    self.match(TRUNCParser.NUM)
                    pass
                elif token in [TRUNCParser.IDV]:
                    self.state = 107
                    self.idd()
                    pass
                elif token in [TRUNCParser.T__16]:
                    self.state = 108
                    self.par()
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


    class VarContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def IDV(self):
            return self.getToken(TRUNCParser.IDV, 0)

        def idd(self):
            return self.getTypedRuleContext(TRUNCParser.IddContext,0)


        def gm(self):
            return self.getTypedRuleContext(TRUNCParser.GmContext,0)


        def getRuleIndex(self):
            return TRUNCParser.RULE_var

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterVar" ):
                listener.enterVar(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitVar" ):
                listener.exitVar(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitVar" ):
                return visitor.visitVar(self)
            else:
                return visitor.visitChildren(self)
            
        def _getText(self, data):
            if not self.idd() is None:
                return self.idd().getVar(data)
            else:
                return self.getText()




    def var(self):

        localctx = TRUNCParser.VarContext(self, self._ctx, self.state)
        self.enterRule(localctx, 22, self.RULE_var)
        try:
            self.state = 116
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,9,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 113
                self.match(TRUNCParser.IDV)
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 114
                self.idd()
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 115
                self.gm()
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
                return self.getTokens(TRUNCParser.IDV)
            else:
                return self.getToken(TRUNCParser.IDV, i)

        def NUM(self):
            return self.getToken(TRUNCParser.NUM, 0)

        def getRuleIndex(self):
            return TRUNCParser.RULE_idd

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
            
        def getVar(self, data):
            if self.IDV(1) is None:
                return self.getText()
            else:
                data_idx = data[self.IDV(1).getText()][0]
                return self.IDV(0).getText()+'['+str(data_idx)+']'  
    
        def getValue(self, data):
            data_name = self.IDV(0).getText()
            if not self.NUM() is None:
                data_idx = int(self.NUM().getText())
            elif not self.IDV(1) is None:
                data_idx = data[self.IDV(1).getText()][0].item()
            return data[data_name][data_idx]




    def idd(self):

        localctx = TRUNCParser.IddContext(self, self._ctx, self.state)
        self.enterRule(localctx, 24, self.RULE_idd)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 118
            self.match(TRUNCParser.IDV)
            self.state = 119
            self.match(TRUNCParser.T__11)
            self.state = 120
            _la = self._input.LA(1)
            if not(_la==TRUNCParser.IDV or _la==TRUNCParser.NUM):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
            self.state = 121
            self.match(TRUNCParser.T__12)
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
                return self.getTypedRuleContexts(TRUNCParser.ListContext)
            else:
                return self.getTypedRuleContext(TRUNCParser.ListContext,i)


        def getRuleIndex(self):
            return TRUNCParser.RULE_gm

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

        localctx = TRUNCParser.GmContext(self, self._ctx, self.state)
        self.enterRule(localctx, 26, self.RULE_gm)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 123
            self.match(TRUNCParser.T__13)
            self.state = 124
            self.list_()
            self.state = 125
            self.match(TRUNCParser.T__14)
            self.state = 126
            self.list_()
            self.state = 127
            self.match(TRUNCParser.T__14)
            self.state = 128
            self.list_()
            self.state = 129
            self.match(TRUNCParser.T__15)
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
                return self.getTokens(TRUNCParser.NUM)
            else:
                return self.getToken(TRUNCParser.NUM, i)

        def par(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(TRUNCParser.ParContext)
            else:
                return self.getTypedRuleContext(TRUNCParser.ParContext,i)


        def getRuleIndex(self):
            return TRUNCParser.RULE_list

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
            
        def unpack(self, params_dict):
            str_list = self.getText()[1:-1].split(',')
            unpacked = torch.zeros(len(str_list))
            for i, elem in enumerate(str_list):
                if elem[0] == '_':
                    unpacked[i] = params_dict[elem[1:]]
                else:
                    unpacked[i] = float(elem)
            return unpacked




    def list_(self):

        localctx = TRUNCParser.ListContext(self, self._ctx, self.state)
        self.enterRule(localctx, 28, self.RULE_list)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 131
            self.match(TRUNCParser.T__11)
            self.state = 134
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [TRUNCParser.NUM]:
                self.state = 132
                self.match(TRUNCParser.NUM)
                pass
            elif token in [TRUNCParser.T__16]:
                self.state = 133
                self.par()
                pass
            else:
                raise NoViableAltException(self)

            self.state = 143
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,12,self._ctx)
            while _alt!=1 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1+1:
                    self.state = 136
                    self.match(TRUNCParser.T__14)
                    self.state = 139
                    self._errHandler.sync(self)
                    token = self._input.LA(1)
                    if token in [TRUNCParser.NUM]:
                        self.state = 137
                        self.match(TRUNCParser.NUM)
                        pass
                    elif token in [TRUNCParser.T__16]:
                        self.state = 138
                        self.par()
                        pass
                    else:
                        raise NoViableAltException(self)
             
                self.state = 145
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,12,self._ctx)

            self.state = 146
            self.match(TRUNCParser.T__12)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class SumContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return TRUNCParser.RULE_sum

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterSum" ):
                listener.enterSum(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitSum" ):
                listener.exitSum(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitSum" ):
                return visitor.visitSum(self)
            else:
                return visitor.visitChildren(self)




    def sum_(self):

        localctx = TRUNCParser.SumContext(self, self._ctx, self.state)
        self.enterRule(localctx, 30, self.RULE_sum)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 148
            self.match(TRUNCParser.T__9)
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
            return TRUNCParser.RULE_sub

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

        localctx = TRUNCParser.SubContext(self, self._ctx, self.state)
        self.enterRule(localctx, 32, self.RULE_sub)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 150
            self.match(TRUNCParser.T__10)
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
            return self.getToken(TRUNCParser.IDV, 0)

        def getRuleIndex(self):
            return TRUNCParser.RULE_par

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
            
        def getValue(self, params_dict):
            name = self.getText()
            name = name[1:]
            return params_dict[name]




    def par(self):

        localctx = TRUNCParser.ParContext(self, self._ctx, self.state)
        self.enterRule(localctx, 34, self.RULE_par)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 152
            self.match(TRUNCParser.T__16)
            self.state = 153
            self.match(TRUNCParser.IDV)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx





