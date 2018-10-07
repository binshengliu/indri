

/*==========================================================================
 * Copyright (c) 2004 University of Massachusetts.  All Rights Reserved.
 *
 * Use of the Lemur Toolkit for Language Modeling and Information Retrieval
 * is subject to the terms of the software license set forth in the LICENSE
 * file included with this software, and also available at
 * http://www.lemurproject.org/license.html
 *
 *==========================================================================
*/

//
// relevancemodel
//
// 23 June 2005 -- tds
//
// Options are:
//    index
//    server
//    query
//    ngram -- default is '1' (unigram)

#include "indri/Parameters.hpp"
#include "indri/RelevanceModel.hpp"
#include "indri/LocalQueryServer.hpp"
#include "indri/ScopedLock.hpp"
#include <queue>

static std::map<std::string, double> parse_field_spec(const std::string& spec);

static bool copy_parameters_to_string_vector( std::vector<std::string>& vec, indri::api::Parameters p, const std::string& parameterName ) {
  if( !p.exists(parameterName) )
    return false;

  indri::api::Parameters slice = p[parameterName];
  
  for( size_t i=0; i<slice.size(); i++ ) {
    vec.push_back( slice[i] );
  }

  return true;
}

static void open_indexes( indri::api::QueryEnvironment& environment, indri::api::Parameters& param ) {
  if( param.exists( "index" ) ) {
    indri::api::Parameters indexes = param["index"];

    for( unsigned int i=0; i < indexes.size(); i++ ) {
      environment.addIndex( std::string(indexes[i]) );
    }
  }

  if( param.exists( "server" ) ) {
    indri::api::Parameters servers = param["server"];

    for( unsigned int i=0; i < servers.size(); i++ ) {
      environment.addServer( std::string(servers[i]) );
    }
  }

  std::vector<std::string> smoothingRules;
  if( copy_parameters_to_string_vector( smoothingRules, param, "rule" ) )
    environment.setScoringRules( smoothingRules );
}

static void usage(indri::api::Parameters param) {
  if (!param.exists("index") || !param.exists("query")
      || !param.exists("fieldB") || !param.exists("fieldWt") || !param.exists("k1")) {
    std::cerr << "bm25f usage: " << std::endl
              << "  bm25f -index=myindex -qno=1 -query=myquery -count=1000 -k1=10 -fieldB=title:8,body:2 -fieldWt=title:6,body:1" << std::endl
              << std::endl;
    exit(-1);
  }
}

struct DocScore {
  struct greater {
    bool operator () (const DocScore &lhs, const DocScore &rhs) const {
      return lhs.score > rhs.score;
    }
  };

  DocScore (lemur::api::DOCID_T id,
            double score):id(id),score(score) { }
  lemur::api::DOCID_T id;
  double score;
};


class DocIterator {
  struct greater {
    bool operator () (indri::index::DocListIterator *lhs, indri::index::DocListIterator *rhs) const {
      return lhs->currentEntry()->document > rhs->currentEntry()->document;
    }
  };
 private:
  std::priority_queue<indri::index::DocListIterator *, vector<indri::index::DocListIterator *>, DocIterator::greater> _docIters;
  indri::index::DocListIterator *_currentIter;
 public:
  DocIterator(indri::index::Index *index, const std::vector<std::string> &stems):
      _currentIter(NULL) {
    for (auto& t: stems) {
      auto *iter = index->docListIterator(t);
      if (iter) {
        iter->startIteration();
        if (!iter->finished()) {
          _docIters.push(iter);
        }
      }
    }
  }

  indri::index::DocListIterator::DocumentData* currentEntry(){
    return _docIters.top()->currentEntry();
  }

  bool nextEntry() {
    if (_docIters.empty()) {
      return false;
    }

    indri::index::DocListIterator *iter = _docIters.top();
    lemur::api::DOCID_T lastId = iter->currentEntry()->document;
    while (!_docIters.empty() && _docIters.top()->currentEntry()->document <= lastId) {
      indri::index::DocListIterator *iter = _docIters.top();
      _docIters.pop();
      if (iter->nextEntry(lastId + 1)) {
        _docIters.push(iter);
      }
    }

    return !_docIters.empty();
  }

  bool finished() {
    return _docIters.empty();
  }
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
 public:
  QueryBM25F(std::string index, std::vector<std::string> fields, std::map<std::string, double> fieldB, std::map<std::string, double> fieldWt, double k1) {
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

  void query(std::string qno, std::string query, int count) {
    std::vector<std::string> stems;
    std::istringstream buf(query);
    std::istream_iterator<std::string> beg(buf), end;
    for (auto t: std::vector<std::string>(beg, end)) {
      stems.push_back(_repo.processTerm(t));
    }

    std::priority_queue<DocScore, vector<DocScore>, DocScore::greater> queue;
    double threshold = 0;
    DocIterator docIters(_index, stems);
    auto dd = docIters.currentEntry();
    while (dd) {
      const indri::index::TermList* tl = _index->termList(dd->document);
      const indri::api::DocumentVector* dv = new indri::api::DocumentVector(_index, tl);

      std::map<std::string, std::map<std::string, int>> termFieldOccur;
      std::map<std::string, int> docFieldLen;
      getFieldInfo(docFieldLen, termFieldOccur, dv, stems);
      delete dv;

      double pseudoFreq = 0;
      double score = 0;
      for (const auto &termPair: termFieldOccur) {
        const std::string &term = termPair.first;
        const std::map<std::string, int> &fieldStats = termPair.second;
        for (const auto &it: fieldStats) {
          const std::string &f = it.first;
          int occ = it.second;

          double fieldFreq = occ / (1 + _fieldB[f] * (docFieldLen[f] / _avgFieldLen[f] - 1));
          pseudoFreq += _fieldWt[f] * fieldFreq;

        }
        double termDocCount = _index->documentCount(term);

        double tf = pseudoFreq / (_k1 + pseudoFreq);

        double idf = (_totalDocumentCount - termDocCount + 0.5) / (termDocCount + 0.5);

        score += tf * idf;
      }

      if (queue.size() < count || score > threshold) {
        queue.push(DocScore(dd->document, score));
        while (queue.size() > count) {
          queue.pop();
        }
        threshold = queue.top().score;
      }

      docIters.nextEntry();
      dd = docIters.currentEntry();
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

  void getFieldInfo(std::map<std::string, int> &docFieldLen,
                    std::map<std::string, std::map<std::string, int>> &termFieldOccur,
                    const indri::api::DocumentVector *dv,
                    const std::vector<std::string> &queryStems) {
    const std::vector<indri::api::DocumentVector::Field>& docFields = dv->fields();
    const std::vector<std::string>& docStems = dv->stems();
    std::map<int, std::string> queryStemPos;
    // Find the indexes of query stems in the document, so we can
    // avoid string comparison when counting the occurrences.
    for (const auto &s: queryStems) {
      size_t i;
      for (i = 0; i < docStems.size(); ++i) {
        if (s == docStems[i]) {
          queryStemPos[i] = s;
          break;
        }
      }
    }

    // Initialize the output map, so we can access the map directly
    // later.
    for (const auto &t: queryStems) {
      termFieldOccur[t] = std::map<std::string, int>();
      for (const auto &f: _fields) {
        termFieldOccur[t][f] = 0;
      }
    }

    for (const auto &f: _fields) {
      docFieldLen[f] = 0;
    }

    const std::vector<int>& positions = dv->positions();
    for (size_t fieldIndex = 0; fieldIndex < docFields.size(); fieldIndex++) {
      const std::string &fieldName = docFields[fieldIndex].name;
      // Not the field we want.
      if (_fields.find(fieldName) == _fields.end()) {
        continue;
      }

      // Accumulate field length
      docFieldLen[fieldName] += docFields[fieldIndex].end - docFields[fieldIndex].begin;

      // Accumulate term field occurrences
      for (size_t pos = docFields[fieldIndex].begin; pos < docFields[fieldIndex].end; ++pos) {
        const std::string &term = queryStemPos[positions[pos]];
        termFieldOccur[term][fieldName] += 1;
      }
    }
  }
};

// open repository
// for each query
//    run query
//    get document vectors from results, save weights from retrieval
//    extract 1-grams, 2-grams, 3-grams etc. as appropriate
//    run background statistics on the n-grams
//    print result
// close repository

int main( int argc, char** argv ) {
  try {
    cerr << "Built with " << INDRI_DISTRIBUTION << endl;

    indri::api::Parameters& param = indri::api::Parameters::instance();
    param.loadCommandLine( argc, argv );
    usage( param );

    std::string index = param["index"];
    std::string query = param["query"];
    std::string qno = param.get("qno", "1");
    int count = param.get("count", 1000);

    int k1 = param.get("k1");
    std::map<std::string, double> fieldB = parse_field_spec(param["fieldB"]);
    std::map<std::string, double> fieldWt = parse_field_spec(param["fieldWt"]);

    // Use a vector to record all the fields
    std::vector<std::string> fields;
    for (auto f: fieldB) {
      fields.push_back(f.first);
    }

    QueryBM25F bm25f(index, fields, fieldB, fieldWt, k1);
    bm25f.query(qno, query, count);
  }
  catch( lemur::api::Exception& e ) {
    LEMUR_ABORT(e);
  } catch( ... ) {
    std::cout << "Caught unhandled exception" << std::endl;
    return -1;
  }

  return 0;
}

static std::map<std::string, double> parse_field_spec(const std::string& spec) {
  std::map<std::string, double> m;

  int nextComma = 0;
  int nextColon = 0;
  int  location = 0;

  for( location = 0; location < spec.length(); ) {
    nextComma = spec.find( ',', location );
    nextColon = spec.find( ':', location );

    std::string key = spec.substr( location, nextColon-location );
    double value = std::stod(spec.substr( nextColon+1, nextComma-nextColon-1 ));

    m[key] = value;

    if( nextComma > 0 )
      location = nextComma+1;
    else
      location = spec.size();
  }

  return m;
}

