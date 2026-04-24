# Generated from SOGA.g4 by ANTLR 4.10.1
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .SOGAParser import SOGAParser
else:
    from SOGAParser import SOGAParser

# This class defines a complete generic visitor for a parse tree produced by SOGAParser.

class SOGAVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by SOGAParser#progr.
    def visitProgr(self, ctx:SOGAParser.ProgrContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#data.
    def visitData(self, ctx:SOGAParser.DataContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#array.
    def visitArray(self, ctx:SOGAParser.ArrayContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#instr.
    def visitInstr(self, ctx:SOGAParser.InstrContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#assignment.
    def visitAssignment(self, ctx:SOGAParser.AssignmentContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#const.
    def visitConst(self, ctx:SOGAParser.ConstContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#const_term.
    def visitConst_term(self, ctx:SOGAParser.Const_termContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#add.
    def visitAdd(self, ctx:SOGAParser.AddContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#add_term.
    def visitAdd_term(self, ctx:SOGAParser.Add_termContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#poly_asgmt.
    def visitPoly_asgmt(self, ctx:SOGAParser.Poly_asgmtContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#poly_aterm.
    def visitPoly_aterm(self, ctx:SOGAParser.Poly_atermContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#poly_mono.
    def visitPoly_mono(self, ctx:SOGAParser.Poly_monoContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#poly_pfactor.
    def visitPoly_pfactor(self, ctx:SOGAParser.Poly_pfactorContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#trig_asgmt.
    def visitTrig_asgmt(self, ctx:SOGAParser.Trig_asgmtContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#trig_aterm.
    def visitTrig_aterm(self, ctx:SOGAParser.Trig_atermContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#trig_mono.
    def visitTrig_mono(self, ctx:SOGAParser.Trig_monoContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#trig_sfactor.
    def visitTrig_sfactor(self, ctx:SOGAParser.Trig_sfactorContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#exp_asgmt.
    def visitExp_asgmt(self, ctx:SOGAParser.Exp_asgmtContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#exp_aterm.
    def visitExp_aterm(self, ctx:SOGAParser.Exp_atermContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#exp_mono.
    def visitExp_mono(self, ctx:SOGAParser.Exp_monoContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#exp_sfactor.
    def visitExp_sfactor(self, ctx:SOGAParser.Exp_sfactorContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#conditional.
    def visitConditional(self, ctx:SOGAParser.ConditionalContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#ifclause.
    def visitIfclause(self, ctx:SOGAParser.IfclauseContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#elseclause.
    def visitElseclause(self, ctx:SOGAParser.ElseclauseContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#block.
    def visitBlock(self, ctx:SOGAParser.BlockContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#bexpr.
    def visitBexpr(self, ctx:SOGAParser.BexprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#lexpr.
    def visitLexpr(self, ctx:SOGAParser.LexprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#monom.
    def visitMonom(self, ctx:SOGAParser.MonomContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#prune.
    def visitPrune(self, ctx:SOGAParser.PruneContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#observe.
    def visitObserve(self, ctx:SOGAParser.ObserveContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#loop.
    def visitLoop(self, ctx:SOGAParser.LoopContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#vars.
    def visitVars(self, ctx:SOGAParser.VarsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#idd.
    def visitIdd(self, ctx:SOGAParser.IddContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#symvars.
    def visitSymvars(self, ctx:SOGAParser.SymvarsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#gm.
    def visitGm(self, ctx:SOGAParser.GmContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#uniform.
    def visitUniform(self, ctx:SOGAParser.UniformContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#list.
    def visitList(self, ctx:SOGAParser.ListContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#par.
    def visitPar(self, ctx:SOGAParser.ParContext):
        return self.visitChildren(ctx)



del SOGAParser