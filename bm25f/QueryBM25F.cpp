#include "indri/Parameters.hpp"
#include "indri/RelevanceModel.hpp"
#include "indri/LocalQueryServer.hpp"
#include "indri/ScopedLock.hpp"
#include "indri/DocExtentListIterator.hpp"
#include "QueryBM25F.hpp"
#include <queue>

bool DocScore::greater::operator () (const DocScore &lhs, const DocScore &rhs) const {
      return lhs.score > rhs.score;
}


bool DocIterator::greater::operator () (indri::index::DocListIterator *lhs, indri::index::DocListIterator *rhs) const {
      return lhs->currentEntry()->document > rhs->currentEntry()->document;
}

bool DocIterator::field_greater::operator () (indri::index::DocExtentListIterator *lhs,
                      indri::index::DocExtentListIterator *rhs) const {
      return lhs->currentEntry()->document > rhs->currentEntry()->document;
}

DocIterator::DocIterator(indri::index::Index *index,
                         const std::set<std::string> &fields,
                         const std::vector<std::string> &stems):
    _currentIter(NULL) {
  for (auto& t: stems) {
    auto *iter = index->docListIterator(t);
    if (iter) {
      iter->startIteration();
      if (!iter->finished()) {
        _termIters.push(iter);
        _termItersMap[t] = iter;
      }
    }
  }

  for (auto& f: fields) {
    auto *iter = index->fieldListIterator(f);
    if (iter) {
      iter->startIteration();
      _fieldIters[f] = iter;
    }
  }
}

DocIterator::entry DocIterator::currentEntry() {
  DocIterator::entry e;
  e.document = _termIters.top()->currentEntry()->document;
  for (auto &termPair: _termItersMap) {
    const std::string &term = termPair.first;
    auto *iter = termPair.second;
    if (!iter->finished() && iter->currentEntry()->document == e.document) {
      e.docEntries[term] = iter->currentEntry();
    }
  }
  for (auto &f: _fieldIters) {
    std::string fieldName = f.first;
    auto *iter = f.second;
    if (!iter->finished() && iter->currentEntry()->document == e.document) {
      e.fieldEntries[fieldName] = iter->currentEntry();
    }
  }

  return e;
}

bool DocIterator::nextEntry() {
  if (nextDocEntry()) {
    nextFieldEntry();
  }

  return true;
}

void DocIterator::nextFieldEntry() {
  auto docEntry = _termIters.top()->currentEntry();

  for (auto &f: _fieldIters) {
    std::string fieldName = f.first;
    indri::index::DocExtentListIterator *fIter = f.second;
    fIter->nextEntry(docEntry->document);
  }

  return;
}

bool DocIterator::nextDocEntry() {
  if (_termIters.empty()) {
    return false;
  }

  indri::index::DocListIterator *iter = _termIters.top();
  lemur::api::DOCID_T lastId = iter->currentEntry()->document;
  while (!_termIters.empty() && _termIters.top()->currentEntry()->document <= lastId) {
    indri::index::DocListIterator *iter = _termIters.top();
    _termIters.pop();
    if (iter->nextEntry(lastId + 1)) {
      _termIters.push(iter);
    }
  }

  return !_termIters.empty();
}

bool DocIterator::finished() {
  return _termIters.empty();
}

QueryBM25F::QueryBM25F(std::string index, std::vector<std::string> fields, std::map<std::string, double> fieldB, std::map<std::string, double> fieldWt, double k1) {
  _repo.openRead(index);

  indri::collection::Repository::index_state state = _repo.indexes();
  _index = (*state)[0];

  _fields = std::set<std::string>(fields.begin(), fields.end());
  _fieldB = fieldB;
  _fieldWt = fieldWt;

  _totalDocumentCount = _index->documentCount();

  // Average field length;
  for (auto f: fields) {
    double len = _index->fieldTermCount(f) / _totalDocumentCount;
    _avgFieldLen[f] = len;
  }

  _k1 = k1;
  _environment.addIndex(index);
};

void QueryBM25F::query(std::string qno, std::string query, int count) {
  std::vector<std::string> stems;
  std::istringstream buf(query);
  std::istream_iterator<std::string> beg(buf), end;
  for (auto t: std::vector<std::string>(beg, end)) {
    stems.push_back(_repo.processTerm(t));
  }

  std::priority_queue<DocScore, vector<DocScore>, DocScore::greater> queue;
  double threshold = 0;
  DocIterator docIters(_index, _fields, stems);
  while (!docIters.finished()) {
    auto de = docIters.currentEntry();
    std::map<std::string, std::map<std::string, int>> termFieldOccur;
    std::map<std::string, int> docFieldLen;
    getFieldInfo(docFieldLen, termFieldOccur, de, stems);

    double pseudoFreq = 0;
    double score = 0;
    for (const auto &termPair: termFieldOccur) {
      const std::string &term = termPair.first;
      const std::map<std::string, int> &fieldStats = termPair.second;
      for (const auto &fieldPair: fieldStats) {
        const std::string &fieldName = fieldPair.first;
        int occurrences = fieldPair.second;

        double fieldFreq = occurrences / (1 + _fieldB[fieldName] * (docFieldLen[fieldName] / _avgFieldLen[fieldName] - 1));
        pseudoFreq += _fieldWt[fieldName] * fieldFreq;

      }
      double termDocCount = _index->documentCount(term);

      double tf = pseudoFreq / (_k1 + pseudoFreq);

      double idf = (_totalDocumentCount - termDocCount + 0.5) / (termDocCount + 0.5);

      score += tf * idf;
    }

    if (queue.size() < count || score > threshold) {
      queue.push(DocScore(de.document, score));
      while (queue.size() > count) {
        queue.pop();
      }
      threshold = queue.top().score;
    }

    docIters.nextEntry();
  }

  std::vector<DocScore> s;
  std::vector<lemur::api::DOCID_T> docids;
  while (queue.size()) {
    DocScore result = queue.top();
    queue.pop();
    s.push_back(result);
    docids.push_back(result.id);
  }

  std::vector<std::string> docnos = _environment.documentMetadata(docids, "docno");

  // 1 Q0 clueweb09-en0007-63-02101 1 -3.34724 indri
  int rank = 1;
  for (int i = s.size() - 1; i >= 0; --i) {
    std::cout << qno << " Q0 " << docnos[i] << " " << rank
              << " " << s[i].score << " " << "bm25f" << std::endl;
    rank += 1;
  }
}

void QueryBM25F::getFieldInfo(std::map<std::string, int> &docFieldLen,
                              std::map<std::string, std::map<std::string, int>> &termFieldOccur,
                              DocIterator::entry &de,
                              const std::vector<std::string> &queryStems) {
  // Initialize the output map, so we can access the map directly
  // later.

  for (auto &fieldPair: de.fieldEntries) {
    const std::string &fieldName = fieldPair.first;
    auto *fieldData = fieldPair.second;
    docFieldLen[fieldName] = 0;
    for (auto &e: fieldData->extents) {
      docFieldLen[fieldName] += e.end - e.begin;
    }
  }

  for (auto &docPair: de.docEntries) {
    const std::string &term = docPair.first;
    auto *docData = docPair.second;

    termFieldOccur[term] = std::map<std::string, int>();

    for (auto &fieldPair: de.fieldEntries) {
      std::string fieldName = fieldPair.first;
      termFieldOccur[term][fieldName] = 0;
      indri::index::DocExtentListIterator::DocumentExtentData *fieldData = fieldPair.second;
      for (auto &e: fieldData->extents) {
        docFieldLen[fieldName] += e.end - e.begin;
        for (auto pos: docData->positions) {
          if (pos >= e.begin && pos < e.end) {
            termFieldOccur[term][fieldName] += 1;
          }
        }
      }
    }
  }
}
