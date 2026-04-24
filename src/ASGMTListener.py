# Generated from ASGMT.g4 by ANTLR 4.10.1
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .ASGMTParser import ASGMTParser
else:
    from ASGMTParser import ASGMTParser

# This class defines a complete listener for a parse tree produced by ASGMTParser.
class ASGMTListener(ParseTreeListener):

    # Enter a parse tree produced by ASGMTParser#assignment.
    def enterAssignment(self, ctx:ASGMTParser.AssignmentContext):
        pass

    # Exit a parse tree produced by ASGMTParser#assignment.
    def exitAssignment(self, ctx:ASGMTParser.AssignmentContext):
        pass


    # Enter a parse tree produced by ASGMTParser#add.
    def enterAdd(self, ctx:ASGMTParser.AddContext):
        pass

    # Exit a parse tree produced by ASGMTParser#add.
    def exitAdd(self, ctx:ASGMTParser.AddContext):
        pass


    # Enter a parse tree produced by ASGMTParser#add_term.
    def enterAdd_term(self, ctx:ASGMTParser.Add_termContext):
        pass

    # Exit a parse tree produced by ASGMTParser#add_term.
    def exitAdd_term(self, ctx:ASGMTParser.Add_termContext):
        pass


    # Enter a parse tree produced by ASGMTParser#term.
    def enterTerm(self, ctx:ASGMTParser.TermContext):
        pass

    # Exit a parse tree produced by ASGMTParser#term.
    def exitTerm(self, ctx:ASGMTParser.TermContext):
        pass


    # Enter a parse tree produced by ASGMTParser#poly_sum.
    def enterPoly_sum(self, ctx:ASGMTParser.Poly_sumContext):
        pass

    # Exit a parse tree produced by ASGMTParser#poly_sum.
    def exitPoly_sum(self, ctx:ASGMTParser.Poly_sumContext):
        pass


    # Enter a parse tree produced by ASGMTParser#poly_term.
    def enterPoly_term(self, ctx:ASGMTParser.Poly_termContext):
        pass

    # Exit a parse tree produced by ASGMTParser#poly_term.
    def exitPoly_term(self, ctx:ASGMTParser.Poly_termContext):
        pass


    # Enter a parse tree produced by ASGMTParser#poly_mono.
    def enterPoly_mono(self, ctx:ASGMTParser.Poly_monoContext):
        pass

    # Exit a parse tree produced by ASGMTParser#poly_mono.
    def exitPoly_mono(self, ctx:ASGMTParser.Poly_monoContext):
        pass


    # Enter a parse tree produced by ASGMTParser#poly_factor.
    def enterPoly_factor(self, ctx:ASGMTParser.Poly_factorContext):
        pass

    # Exit a parse tree produced by ASGMTParser#poly_factor.
    def exitPoly_factor(self, ctx:ASGMTParser.Poly_factorContext):
        pass


    # Enter a parse tree produced by ASGMTParser#trig_sum.
    def enterTrig_sum(self, ctx:ASGMTParser.Trig_sumContext):
        pass

    # Exit a parse tree produced by ASGMTParser#trig_sum.
    def exitTrig_sum(self, ctx:ASGMTParser.Trig_sumContext):
        pass


    # Enter a parse tree produced by ASGMTParser#trig_term.
    def enterTrig_term(self, ctx:ASGMTParser.Trig_termContext):
        pass

    # Exit a parse tree produced by ASGMTParser#trig_term.
    def exitTrig_term(self, ctx:ASGMTParser.Trig_termContext):
        pass


    # Enter a parse tree produced by ASGMTParser#trig_mono.
    def enterTrig_mono(self, ctx:ASGMTParser.Trig_monoContext):
        pass

    # Exit a parse tree produced by ASGMTParser#trig_mono.
    def exitTrig_mono(self, ctx:ASGMTParser.Trig_monoContext):
        pass


    # Enter a parse tree produced by ASGMTParser#trig_factor.
    def enterTrig_factor(self, ctx:ASGMTParser.Trig_factorContext):
        pass

    # Exit a parse tree produced by ASGMTParser#trig_factor.
    def exitTrig_factor(self, ctx:ASGMTParser.Trig_factorContext):
        pass


    # Enter a parse tree produced by ASGMTParser#exp_sum.
    def enterExp_sum(self, ctx:ASGMTParser.Exp_sumContext):
        pass

    # Exit a parse tree produced by ASGMTParser#exp_sum.
    def exitExp_sum(self, ctx:ASGMTParser.Exp_sumContext):
        pass


    # Enter a parse tree produced by ASGMTParser#exp_term.
    def enterExp_term(self, ctx:ASGMTParser.Exp_termContext):
        pass

    # Exit a parse tree produced by ASGMTParser#exp_term.
    def exitExp_term(self, ctx:ASGMTParser.Exp_termContext):
        pass


    # Enter a parse tree produced by ASGMTParser#exp_mono.
    def enterExp_mono(self, ctx:ASGMTParser.Exp_monoContext):
        pass

    # Exit a parse tree produced by ASGMTParser#exp_mono.
    def exitExp_mono(self, ctx:ASGMTParser.Exp_monoContext):
        pass


    # Enter a parse tree produced by ASGMTParser#exp_factor.
    def enterExp_factor(self, ctx:ASGMTParser.Exp_factorContext):
        pass

    # Exit a parse tree produced by ASGMTParser#exp_factor.
    def exitExp_factor(self, ctx:ASGMTParser.Exp_factorContext):
        pass


    # Enter a parse tree produced by ASGMTParser#symvars.
    def enterSymvars(self, ctx:ASGMTParser.SymvarsContext):
        pass

    # Exit a parse tree produced by ASGMTParser#symvars.
    def exitSymvars(self, ctx:ASGMTParser.SymvarsContext):
        pass


    # Enter a parse tree produced by ASGMTParser#idd.
    def enterIdd(self, ctx:ASGMTParser.IddContext):
        pass

    # Exit a parse tree produced by ASGMTParser#idd.
    def exitIdd(self, ctx:ASGMTParser.IddContext):
        pass


    # Enter a parse tree produced by ASGMTParser#gm.
    def enterGm(self, ctx:ASGMTParser.GmContext):
        pass

    # Exit a parse tree produced by ASGMTParser#gm.
    def exitGm(self, ctx:ASGMTParser.GmContext):
        pass


    # Enter a parse tree produced by ASGMTParser#list.
    def enterList(self, ctx:ASGMTParser.ListContext):
        pass

    # Exit a parse tree produced by ASGMTParser#list.
    def exitList(self, ctx:ASGMTParser.ListContext):
        pass


    # Enter a parse tree produced by ASGMTParser#sub.
    def enterSub(self, ctx:ASGMTParser.SubContext):
        pass

    # Exit a parse tree produced by ASGMTParser#sub.
    def exitSub(self, ctx:ASGMTParser.SubContext):
        pass


    # Enter a parse tree produced by ASGMTParser#par.
    def enterPar(self, ctx:ASGMTParser.ParContext):
        pass

    # Exit a parse tree produced by ASGMTParser#par.
    def exitPar(self, ctx:ASGMTParser.ParContext):
        pass



del ASGMTParser