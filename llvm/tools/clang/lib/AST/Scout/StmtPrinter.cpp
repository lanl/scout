// Note - this file is included by the StmtPrinter source file 
// one directory up (StmtPrinter is all contained in a single
// file there...). 
//  

void StmtPrinter::VisitForAllStmt(ForAllStmt *Node) 
{ }

void StmtPrinter::VisitForAllArrayStmt(ForAllArrayStmt *Node) 
{ }

void StmtPrinter::VisitRenderAllStmt(RenderAllStmt *Node) 
{ }

void StmtPrinter::VisitVolumeRenderAllStmt(VolumeRenderAllStmt *Node) 
{ }

void StmtPrinter::VisitScoutVectorMemberExpr(ScoutVectorMemberExpr *Node) {

  bool isColor = false;

  if (DeclRefExpr* dr = dyn_cast<DeclRefExpr>(Node->getBase())) {
    if (dr->getDecl()->getName() == "color") {
      isColor = true;
    }
  }

  PrintExpr(Node->getBase());

  OS << ".";
  switch(Node->getIdx()) {
    case 0:
      if (isColor) {
        OS << "r";
      } else {
        OS << "x";
      }
      break;

    case 1:
      if (isColor) {
        OS << "g";
      } else {
        OS << "y";
      }
      break;

    case 2:
      if (isColor) {
        OS << "b";
      } else {
        OS << "z";
      }
      break;

    case 3:
      if (isColor) {
        OS << "a";
      } else {
        OS << "w";
      }
      break;
  }
}

