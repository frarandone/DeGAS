# Generated from ASGMT.g4 by ANTLR 4.10.1
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .ASGMTParser import ASGMTParser
else:
    from ASGMTParser import ASGMTParser

# This class defines a complete generic visitor for a parse tree produced by ASGMTParser.

class ASGMTVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by ASGMTParser#assignment.
    def visitAssignment(self, ctx:ASGMTParser.AssignmentContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ASGMTParser#add.
    def visitAdd(self, ctx:ASGMTParser.AddContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ASGMTParser#add_term.
    def visitAdd_term(self, ctx:ASGMTParser.Add_termContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ASGMTParser#term.
    def visitTerm(self, ctx:ASGMTParser.TermContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ASGMTParser#poly_sum.
    def visitPoly_sum(self, ctx:ASGMTParser.Poly_sumContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ASGMTParser#poly_term.
    def visitPoly_term(self, ctx:ASGMTParser.Poly_termContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ASGMTParser#poly_mono.
    def visitPoly_mono(self, ctx:ASGMTParser.Poly_monoContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ASGMTParser#poly_factor.
    def visitPoly_factor(self, ctx:ASGMTParser.Poly_factorContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ASGMTParser#trig_sum.
    def visitTrig_sum(self, ctx:ASGMTParser.Trig_sumContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ASGMTParser#trig_term.
    def visitTrig_term(self, ctx:ASGMTParser.Trig_termContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ASGMTParser#trig_mono.
    def visitTrig_mono(self, ctx:ASGMTParser.Trig_monoContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ASGMTParser#trig_factor.
    def visitTrig_factor(self, ctx:ASGMTParser.Trig_factorContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ASGMTParser#exp_sum.
    def visitExp_sum(self, ctx:ASGMTParser.Exp_sumContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ASGMTParser#exp_term.
    def visitExp_term(self, ctx:ASGMTParser.Exp_termContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ASGMTParser#exp_mono.
    def visitExp_mono(self, ctx:ASGMTParser.Exp_monoContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ASGMTParser#exp_factor.
    def visitExp_factor(self, ctx:ASGMTParser.Exp_factorContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ASGMTParser#symvars.
    def visitSymvars(self, ctx:ASGMTParser.SymvarsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ASGMTParser#idd.
    def visitIdd(self, ctx:ASGMTParser.IddContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ASGMTParser#gm.
    def visitGm(self, ctx:ASGMTParser.GmContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ASGMTParser#list.
    def visitList(self, ctx:ASGMTParser.ListContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ASGMTParser#sub.
    def visitSub(self, ctx:ASGMTParser.SubContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ASGMTParser#par.
    def visitPar(self, ctx:ASGMTParser.ParContext):
        return self.visitChildren(ctx)



del ASGMTParser