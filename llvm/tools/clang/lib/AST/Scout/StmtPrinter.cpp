// Note - this file is included by the StmtPrinter source file 
// one directory up (StmtPrinter is all contained in a single
// file there...). 
//  
void StmtPrinter::VisitForallMeshStmt(ForallMeshStmt *Node) {
  Indent() << "forall ";

  if (Node->isOverCells())
    OS << "cells ";
  else if (Node->isOverEdges())
    OS << "edges ";
  else if (Node->isOverVertices())
    OS << "verticies ";
  else if (Node->isOverFaces())
    OS << "faces ";
  else 
    OS << "<unknown mesh element>";

  OS << Node->getRefVarInfo()->getName() << " in ";
  OS << Node->getMeshInfo()->getName() << " ";

  if (Node->hasPredicate()) {
    OS << "(";
    PrintExpr(Node->getPredicate());
    OS << ")";
  }

  if (CompoundStmt *CS = dyn_cast<CompoundStmt>(Node->getBody())) {
    PrintRawCompoundStmt(CS);
    OS << "\n";
  } else {
    OS << "\n";
    PrintStmt(Node->getBody());
  }
}

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

