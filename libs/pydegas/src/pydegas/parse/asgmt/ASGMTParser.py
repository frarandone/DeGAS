# Generated from ASGMT.g4 by ANTLR 4.10.1
# encoding: utf-8
from antlr4 import *
from io import StringIO
import sys
if sys.version_info[1] > 5:
	from typing import TextIO
else:
	from typing.io import TextIO

def serializedATN():
    return [
        4,1,15,98,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,6,7,
        6,2,7,7,7,2,8,7,8,2,9,7,9,1,0,1,0,1,0,1,0,1,0,1,1,1,1,3,1,28,8,1,
        1,1,5,1,31,8,1,10,1,12,1,34,9,1,1,2,1,2,1,2,5,2,39,8,2,10,2,12,2,
        42,9,2,1,3,1,3,3,3,46,8,3,1,3,1,3,3,3,50,8,3,1,3,1,3,3,3,54,8,3,
        1,3,3,3,57,8,3,1,4,1,4,3,4,61,8,4,1,5,1,5,1,5,1,5,1,5,1,6,1,6,1,
        6,1,6,1,6,1,6,1,6,1,6,1,7,1,7,1,7,3,7,79,8,7,1,7,1,7,1,7,3,7,84,
        8,7,5,7,86,8,7,10,7,12,7,89,9,7,1,7,1,7,1,8,1,8,1,9,1,9,1,9,1,9,
        2,32,87,0,10,0,2,4,6,8,10,12,14,16,18,0,1,1,0,11,12,100,0,20,1,0,
        0,0,2,25,1,0,0,0,4,35,1,0,0,0,6,56,1,0,0,0,8,60,1,0,0,0,10,62,1,
        0,0,0,12,67,1,0,0,0,14,75,1,0,0,0,16,92,1,0,0,0,18,94,1,0,0,0,20,
        21,3,8,4,0,21,22,5,1,0,0,22,23,3,2,1,0,23,24,5,0,0,1,24,1,1,0,0,
        0,25,32,3,4,2,0,26,28,5,2,0,0,27,26,1,0,0,0,27,28,1,0,0,0,28,29,
        1,0,0,0,29,31,3,4,2,0,30,27,1,0,0,0,31,34,1,0,0,0,32,33,1,0,0,0,
        32,30,1,0,0,0,33,3,1,0,0,0,34,32,1,0,0,0,35,40,3,6,3,0,36,37,5,3,
        0,0,37,39,3,6,3,0,38,36,1,0,0,0,39,42,1,0,0,0,40,38,1,0,0,0,40,41,
        1,0,0,0,41,5,1,0,0,0,42,40,1,0,0,0,43,57,5,12,0,0,44,46,3,16,8,0,
        45,44,1,0,0,0,45,46,1,0,0,0,46,47,1,0,0,0,47,57,3,18,9,0,48,50,3,
        16,8,0,49,48,1,0,0,0,49,50,1,0,0,0,50,51,1,0,0,0,51,57,3,8,4,0,52,
        54,3,16,8,0,53,52,1,0,0,0,53,54,1,0,0,0,54,55,1,0,0,0,55,57,3,12,
        6,0,56,43,1,0,0,0,56,45,1,0,0,0,56,49,1,0,0,0,56,53,1,0,0,0,57,7,
        1,0,0,0,58,61,5,11,0,0,59,61,3,10,5,0,60,58,1,0,0,0,60,59,1,0,0,
        0,61,9,1,0,0,0,62,63,5,11,0,0,63,64,5,4,0,0,64,65,7,0,0,0,65,66,
        5,5,0,0,66,11,1,0,0,0,67,68,5,6,0,0,68,69,3,14,7,0,69,70,5,7,0,0,
        70,71,3,14,7,0,71,72,5,7,0,0,72,73,3,14,7,0,73,74,5,8,0,0,74,13,
        1,0,0,0,75,78,5,4,0,0,76,79,5,12,0,0,77,79,3,18,9,0,78,76,1,0,0,
        0,78,77,1,0,0,0,79,87,1,0,0,0,80,83,5,7,0,0,81,84,5,12,0,0,82,84,
        3,18,9,0,83,81,1,0,0,0,83,82,1,0,0,0,84,86,1,0,0,0,85,80,1,0,0,0,
        86,89,1,0,0,0,87,88,1,0,0,0,87,85,1,0,0,0,88,90,1,0,0,0,89,87,1,
        0,0,0,90,91,5,5,0,0,91,15,1,0,0,0,92,93,5,9,0,0,93,17,1,0,0,0,94,
        95,5,10,0,0,95,96,5,11,0,0,96,19,1,0,0,0,11,27,32,40,45,49,53,56,
        60,78,83,87
    ]

class ASGMTParser ( Parser ):

    grammarFileName = "ASGMT.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    sharedContextCache = PredictionContextCache()

    literalNames = [ "<INVALID>", "'='", "'+'", "'*'", "'['", "']'", "'gm('",
                     "','", "')'", "'-'", "'_'" ]

    symbolicNames = [ "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>",
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>",
                      "<INVALID>", "<INVALID>", "<INVALID>", "IDV", "NUM",
                      "COMM", "WS", "DIGIT" ]

    RULE_assignment = 0
    RULE_add = 1
    RULE_add_term = 2
    RULE_term = 3
    RULE_symvars = 4
    RULE_idd = 5
    RULE_gm = 6
    RULE_list = 7
    RULE_sub = 8
    RULE_par = 9

    ruleNames =  [ "assignment", "add", "add_term", "term", "symvars", "idd",
                   "gm", "list", "sub", "par" ]

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
    IDV=11
    NUM=12
    COMM=13
    WS=14
    DIGIT=15

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


        def EOF(self):
            return self.getToken(ASGMTParser.EOF, 0)

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
            self.state = 20
            self.symvars()
            self.state = 21
            self.match(ASGMTParser.T__0)
            self.state = 22
            self.add()
            self.state = 23
            self.match(ASGMTParser.EOF)
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
            self.state = 25
            self.add_term()
            self.state = 32
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,1,self._ctx)
            while _alt!=1 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1+1:
                    self.state = 27
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if _la==ASGMTParser.T__1:
                        self.state = 26
                        self.match(ASGMTParser.T__1)


                    self.state = 29
                    self.add_term()
                self.state = 34
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,1,self._ctx)

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
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 35
            self.term()
            self.state = 40
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==ASGMTParser.T__2:
                self.state = 36
                self.match(ASGMTParser.T__2)
                self.state = 37
                self.term()
                self.state = 42
                self._errHandler.sync(self)
                _la = self._input.LA(1)

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
            self.state = 56
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,6,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 43
                self.match(ASGMTParser.NUM)
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 45
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==ASGMTParser.T__8:
                    self.state = 44
                    self.sub()


                self.state = 47
                self.par()
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 49
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==ASGMTParser.T__8:
                    self.state = 48
                    self.sub()


                self.state = 51
                self.symvars()
                pass

            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 53
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==ASGMTParser.T__8:
                    self.state = 52
                    self.sub()


                self.state = 55
                self.gm()
                pass


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
        self.enterRule(localctx, 8, self.RULE_symvars)
        try:
            self.state = 60
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,7,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 58
                self.match(ASGMTParser.IDV)
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 59
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
        self.enterRule(localctx, 10, self.RULE_idd)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 62
            self.match(ASGMTParser.IDV)
            self.state = 63
            self.match(ASGMTParser.T__3)
            self.state = 64
            _la = self._input.LA(1)
            if not(_la==ASGMTParser.IDV or _la==ASGMTParser.NUM):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
            self.state = 65
            self.match(ASGMTParser.T__4)
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
        self.enterRule(localctx, 12, self.RULE_gm)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 67
            self.match(ASGMTParser.T__5)
            self.state = 68
            self.list_()
            self.state = 69
            self.match(ASGMTParser.T__6)
            self.state = 70
            self.list_()
            self.state = 71
            self.match(ASGMTParser.T__6)
            self.state = 72
            self.list_()
            self.state = 73
            self.match(ASGMTParser.T__7)
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
        self.enterRule(localctx, 14, self.RULE_list)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 75
            self.match(ASGMTParser.T__3)
            self.state = 78
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [ASGMTParser.NUM]:
                self.state = 76
                self.match(ASGMTParser.NUM)
                pass
            elif token in [ASGMTParser.T__9]:
                self.state = 77
                self.par()
                pass
            else:
                raise NoViableAltException(self)

            self.state = 87
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,10,self._ctx)
            while _alt!=1 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1+1:
                    self.state = 80
                    self.match(ASGMTParser.T__6)
                    self.state = 83
                    self._errHandler.sync(self)
                    token = self._input.LA(1)
                    if token in [ASGMTParser.NUM]:
                        self.state = 81
                        self.match(ASGMTParser.NUM)
                        pass
                    elif token in [ASGMTParser.T__9]:
                        self.state = 82
                        self.par()
                        pass
                    else:
                        raise NoViableAltException(self)

                self.state = 89
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,10,self._ctx)

            self.state = 90
            self.match(ASGMTParser.T__4)
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
        self.enterRule(localctx, 16, self.RULE_sub)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 92
            self.match(ASGMTParser.T__8)
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
        self.enterRule(localctx, 18, self.RULE_par)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 94
            self.match(ASGMTParser.T__9)
            self.state = 95
            self.match(ASGMTParser.IDV)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx
