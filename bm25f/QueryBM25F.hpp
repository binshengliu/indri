#include "indri/Parameters.hpp"
#include "indri/LocalQueryServer.hpp"
#include "indri/ScopedLock.hpp"
#include "indri/DocExtentListIterator.hpp"
#include "indri/QueryEnvironment.hpp"
#include <queue>

struct DocScore {
  struct greater {
    bool operator () (const DocScore &lhs, const DocScore &rhs) const;
  };

  DocScore (lemur::api::DOCID_T id,
            double score):id(id),score(score) { }
  lemur::api::DOCID_T id;
  double score;
};


class DocIterator {
  struct greater {
    bool operator () (indri::index::DocListIterator *lhs, indri::index::DocListIterator *rhs) const;
  };

  struct field_greater {
    bool operator () (indri::index::DocExtentListIterator *lhs,
                      indri::index::DocExtentListIterator *rhs) const;
  };

 public:
  struct entry {
    lemur::api::DOCID_T document;
    std::map<std::string, indri::index::DocListIterator::DocumentData *> docEntries;
    std::map<std::string, indri::index::DocExtentListIterator::DocumentExtentData *> fieldEntries;
  };
 private:
  std::map<std::string, indri::index::DocExtentListIterator *> _fieldIters;
  std::priority_queue<indri::index::DocListIterator *, vector<indri::index::DocListIterator *>, DocIterator::greater> _termIters;
  std::map<std::string, indri::index::DocListIterator *> _termItersMap;
  indri::index::DocListIterator *_currentIter;
 public:
  DocIterator(indri::index::Index *index,
              const std::set<std::string> &fields,
              const std::vector<std::string> &stems);
  DocIterator::entry currentEntry();
  bool nextEntry();
  void nextFieldEntry();
  bool nextDocEntry();
  bool finished();
};

class QueryBM25F {
 private:
  indri::collection::Repository _repo;
  indri::index::Index *_index;
  std::set<std::string> _fields;
  std::map<std::string, double> _fieldB;
  std::map<std::string, double> _fieldWt;
  double _totalDocumentCount;
  std::map<std::string, double> _avgFieldLen;
  double _k1;
  indri::api::QueryEnvironment _environment;
  int _requested;
 public:
  QueryBM25F(std::string index, std::vector<std::string> fields, std::map<std::string, double> fieldB, std::map<std::string, double> fieldWt, double k1, int requested);

  void query(std::string qno, std::string query);

  void getFieldInfo(std::map<std::string, int> &docFieldLen,
                    std::map<std::string, std::map<std::string, int>> &termFieldOccur,
                    DocIterator::entry &de,
                    const std::vector<std::string> &queryStems);
};

