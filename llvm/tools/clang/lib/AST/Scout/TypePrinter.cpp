// Note - this file is included by the TypePrinter source file 
// one directory up (TypePrinter is all contained in a single
// file there...). 
//  
void
TypePrinter::printUniformMeshBefore(const UniformMeshType *T,
                                    raw_ostream &OS) 
{
  MeshDecl* MD = T->getDecl();
  OS << MD->getIdentifier()->getName().str();
}

void
TypePrinter::printStructuredMeshBefore(const StructuredMeshType *T,
                                       raw_ostream &OS)
{
  MeshDecl* MD = T->getDecl();
  OS << MD->getIdentifier()->getName().str();
}


void
TypePrinter::printRectilinearMeshBefore(const RectilinearMeshType *T,
                                        raw_ostream &OS)
{
  MeshDecl* MD = T->getDecl();
  OS << MD->getIdentifier()->getName().str();
}


void 
TypePrinter::printUnstructuredMeshBefore(const UnstructuredMeshType *T, 
                                         raw_ostream &OS) 
{
  MeshDecl* MD = T->getDecl();
  OS << MD->getIdentifier()->getName().str();
}


void
TypePrinter::printUniformMeshAfter(const UniformMeshType *T,
                                   raw_ostream &OS)
{
  OS << '[';
  MeshType::MeshDimensions dv = T->dimensions();
  MeshType::MeshDimensions::iterator dimiter;
  for (dimiter = dv.begin();
      dimiter != dv.end();
      dimiter++){
    (*dimiter)->printPretty(OS, 0, Policy);
    if (dimiter+1 != dv.end()) OS << ',';
  }
  OS << ']';
}

void
TypePrinter::printStructuredMeshAfter(const StructuredMeshType *T,
                                      raw_ostream &OS)
{ }


void
TypePrinter::printRectilinearMeshAfter(const RectilinearMeshType *T,
                                      raw_ostream &OS)
{ }


void
TypePrinter::printUnstructuredMeshAfter(const UnstructuredMeshType *T,
                                        raw_ostream &OS)
{ }
