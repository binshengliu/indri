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

DocIterator::entry::entry(lemur::api::DOCID_T document,
                          size_t termCount,
                          size_t fieldCount):
    document(document),
    docEntries(termCount, NULL),
    fieldEntries(fieldCount, NULL) {}

DocIterator::DocIterator(indri::index::Index *index,
                         const std::vector<std::string> &fields,
                         const std::vector<std::string> &stems):
    _currentIter(NULL),
    _termItersMap(stems.size(), NULL),
    _fieldIters(fields.size(), NULL) {
  for (size_t termIndex = 0; termIndex < stems.size(); ++termIndex) {
    auto *iter = index->docListIterator(stems[termIndex]);
    if (iter) {
      iter->startIteration();
      if (!iter->finished()) {
        _termIters.push(iter);
        _termItersMap[termIndex] = iter;
      }
    }
  }

  for (size_t fieldIndex = 0; fieldIndex < fields.size(); ++fieldIndex) {
    auto *iter = index->fieldListIterator(fields[fieldIndex]);
    if (iter) {
      iter->startIteration();
      if (!iter->finished()) {
        _fieldIters[fieldIndex] = iter;
      }
    }
  }
}

DocIterator::entry DocIterator::currentEntry() {
  lemur::api::DOCID_T document = _termIters.top()->currentEntry()->document;
  DocIterator::entry e(document, _termItersMap.size(), _fieldIters.size());
  for (size_t termIndex = 0; termIndex < _termItersMap.size(); ++termIndex) {
    auto iter = _termItersMap[termIndex];
    if (!iter->finished() && iter->currentEntry()->document == e.document) {
      e.docEntries[termIndex] = iter->currentEntry();
    }
  }

  for (size_t fieldIndex = 0; fieldIndex < _fieldIters.size(); ++fieldIndex) {
    auto iter = _fieldIters[fieldIndex];
    if (!iter->finished() && iter->currentEntry()->document == e.document) {
      e.fieldEntries[fieldIndex] = iter->currentEntry();
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

  for (auto &iter: _fieldIters) {
    iter->nextEntry(docEntry->document);
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

QueryBM25F::QueryBM25F(std::string index,
                       std::vector<std::string> fields,
                       std::vector<double> fieldB,
                       std::vector<double> fieldWt,
                       double k1,
                       int requested):
    _fields(fields),
    _fieldB(fieldB),
    _fieldWt(fieldWt),
    _k1(k1),
    _requested(requested),
    _avgFieldLen(fields.size(), 0)
{
  _repo.openRead(index);

  indri::collection::Repository::index_state state = _repo.indexes();
  _index = (*state)[0];
  _totalDocumentCount = _index->documentCount();

  // Average field length;
  for (size_t fieldIndex = 0; fieldIndex < fields.size(); ++fieldIndex) {
    double len = _index->fieldTermCount(fields[fieldIndex]) / _totalDocumentCount;
    _avgFieldLen[fieldIndex] = len;
  }

  _environment.addIndex(index);
};

void QueryBM25F::query(std::string qno, std::string query) {
  std::vector<std::string> stems;
  std::istringstream buf(query);
  std::istream_iterator<std::string> beg(buf), end;
  for (auto t: std::vector<std::string>(beg, end)) {
    stems.push_back(_repo.processTerm(t));
  }

  std::vector<double> termDocCounts(stems.size(), 0);
  for (size_t termIndex = 0; termIndex < stems.size(); ++termIndex) {
    double count = _index->documentCount(stems[termIndex]);
    termDocCounts[termIndex] = count;
  }

  std::priority_queue<DocScore, vector<DocScore>, DocScore::greater> queue;
  double threshold = 0;
  DocIterator docIters(_index, _fields, stems);
  while (!docIters.finished()) {
    auto de = docIters.currentEntry();
    std::vector<std::vector<int>> termFieldOccur(stems.size(),
                                                 std::vector<int>(_fields.size(), 0));
    std::vector<int> docFieldLen(_fields.size(), 0);

    getFieldInfo(docFieldLen, termFieldOccur, de, stems);

    double pseudoFreq = 0;
    double score = 0;
    for (size_t termIndex = 0; termIndex < stems.size(); ++termIndex) {
      const std::vector<int> &fieldStats = termFieldOccur[termIndex];
      for (size_t fieldIndex = 0; fieldIndex < _fields.size(); ++fieldIndex) {
        int occurrences = fieldStats[fieldIndex];

        double fieldFreq = occurrences / (1 + _fieldB[fieldIndex] * (docFieldLen[fieldIndex] / _avgFieldLen[fieldIndex] - 1));
        pseudoFreq += _fieldWt[fieldIndex] * fieldFreq;

      }
      double tf = pseudoFreq / (_k1 + pseudoFreq);

      double termDocCount = termDocCounts[termIndex];
      double idf = (_totalDocumentCount - termDocCount + 0.5) / (termDocCount + 0.5);

      score += tf * idf;
    }

    if (queue.size() < _requested || score > threshold) {
      queue.push(DocScore(de.document, score));
      while (queue.size() > _requested) {
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

void QueryBM25F::getFieldInfo(std::vector<int> &docFieldLen,
                              std::vector<std::vector<int>> &termFieldOccur,
                              DocIterator::entry &de,
                              const std::vector<std::string> &queryStems) {
  // Initialize the output map, so we can access the map directly
  // later.

  for (size_t fieldIndex = 0; fieldIndex < de.fieldEntries.size(); ++fieldIndex) {
    auto *fieldData = de.fieldEntries[fieldIndex];
    if (!fieldData) {
      continue;
    }
    for (auto &e: fieldData->extents) {
      docFieldLen[fieldIndex] += e.end - e.begin;
    }
  }

  for (size_t termIndex = 0; termIndex < de.docEntries.size(); ++termIndex) {
    auto *docData = de.docEntries[termIndex];
    if (!docData) {
      continue;
    }

    for (size_t fieldIndex = 0; fieldIndex < _fields.size(); ++fieldIndex) {
      indri::index::DocExtentListIterator::DocumentExtentData *fieldData = de.fieldEntries[fieldIndex];
      if (!fieldData) {
        continue;
      }
      for (auto &e: fieldData->extents) {
        for (auto pos: docData->positions) {
          if (pos >= e.begin && pos < e.end) {
            termFieldOccur[termIndex][fieldIndex] += 1;
          }
        }
      }
    }
  }
}
