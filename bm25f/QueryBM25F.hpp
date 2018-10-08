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
    std::vector<indri::index::DocListIterator::DocumentData *> docEntries;
    std::vector<indri::index::DocExtentListIterator::DocumentExtentData *> fieldEntries;
    entry(lemur::api::DOCID_T document, size_t termCount, size_t fieldCount);
  };
 private:
  std::vector<indri::index::DocListIterator *> _termIters;
  std::vector<indri::index::DocExtentListIterator *> _fieldIters;
  std::priority_queue<indri::index::DocListIterator *, vector<indri::index::DocListIterator *>, DocIterator::greater> _termItersQueue;
  indri::index::DocListIterator *_currentIter;
 public:
  DocIterator(indri::index::Index *index,
              const std::vector<std::string> &fields,
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
  std::vector<std::string> _fields;
  std::vector<double> _fieldB;
  std::vector<double> _fieldWt;
  double _totalDocumentCount;
  std::vector<double> _avgFieldLen;
  double _k1;
  indri::api::QueryEnvironment _environment;
  int _requested;
 public:
  QueryBM25F(std::string index,
             std::vector<std::string> fields,
             std::vector<double> fieldB,
             std::vector<double> fieldWt,
             double k1,
             int requested);

  void query(std::string qno, std::string query);

  void getFieldInfo(std::vector<int> &docFieldLen,
                    std::vector<std::vector<int>> &termFieldOccur,
                    DocIterator::entry &de,
                    const std::vector<std::string> &queryStems);
};

