

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

static void parse_field( std::map<std::string, double> &m, const std::string& spec );

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
  if (!param.exists("index") || !param.exists("run")) {
    std::cerr << "run_field usage: " << std::endl
              << "  doc_text -index=myindex -field=myfield -run=myrun" << std::endl
              << "     myfield: a valid field in the index, or \"all\" for whole document" << std::endl
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


class QueryBM25F {
 private:
  indri::collection::Repository _repo;
  indri::index::Index *_index;
  std::vector<indri::index::DocListIterator *> _lists;
  std::set<std::string> _fieldSet;
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

    _fieldSet = std::set<std::string>(fields.begin(), fields.end());
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
    std::vector<std::string> terms;
    std::istringstream buf(query);
    std::istream_iterator<std::string> beg(buf), end;
    for (auto t: std::vector<std::string>(beg, end)) {
      terms.push_back(_repo.processTerm(t));
    }

    for (auto& t: terms) {
      auto *iter = _index->docListIterator(t);
      if (iter) {
        iter->startIteration();
        _lists.push_back(iter);
      }
    }

    std::priority_queue<DocScore, vector<DocScore>, DocScore::greater> queue;
    double threshold = 0;
    for (auto dd = next(); dd; dd = next()) {
      const indri::index::TermList* tl = _index->termList(dd->document);
      indri::api::DocumentVector* dv = new indri::api::DocumentVector(_index, tl);

      std::vector<indri::api::DocumentVector::Field>& fields = dv->fields();
      std::vector<std::string>& stems = dv->stems();
      const std::vector<int>& positions = dv->positions();
      std::map<std::string, std::map<std::string, int>> fOcc;
      std::map<std::string, int> docFieldLen;
      getFieldInfo(docFieldLen, fOcc, dv, terms);
      delete dv;

      double pseudoFreq = 0;
      double score = 0;
      for (auto termPair: fOcc) {
        std::string term = termPair.first;
        std::map<std::string, int> fieldStats = termPair.second;
        for (auto it: fieldStats) {
          std::string f = it.first;
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
                    std::map<std::string, std::map<std::string, int>> &fOcc,
                    indri::api::DocumentVector *doc,
                    std::vector<std::string> queryStems) {
    std::vector<indri::api::DocumentVector::Field>& fields = doc->fields();
    std::vector<std::string>& stems = doc->stems();
    std::vector<int> queryStemPos;
    for (auto &s: queryStems) {
      size_t i;
      for (i = 0; i < stems.size(); ++i) {
        if (s == stems[i]) {
          queryStemPos.push_back(i);
        }
      }

      // Not found
      if (i == stems.size()) {
        queryStemPos.push_back(-1);
      }
    }

    const std::vector<int>& positions = doc->positions();
    for (size_t f = 0; f < fields.size(); f++) {
      std::string fn = fields[f].name;
      // Not the field we want.
      if (_fieldSet.find(fn) == _fieldSet.end()) {
        continue;
      }

      // Accumulate field length
      if (docFieldLen.find(fn) == docFieldLen.end()) {
        docFieldLen[fn] = 0;
      }
      docFieldLen[fn] += fields[f].end - fields[f].begin;

      // Accumulate term field occurrences
      size_t count = 0;
      for (size_t pos = fields[f].begin; pos < fields[f].end; ++pos) {
        for (size_t i = 0; i < queryStemPos.size(); ++i) {
          if (positions[pos] == queryStemPos[i]) {
            if (fOcc.find(queryStems[i]) == fOcc.end()) {
              fOcc[queryStems[i]] = std::map<std::string, int>();
            }
            if (fOcc[queryStems[i]].find(fields[f].name) == fOcc[queryStems[i]].end()) {
              fOcc[queryStems[i]][fields[f].name] = 0;
            }
            fOcc[queryStems[i]][fields[f].name] += 1;
          }
        }
      }
    }

  }

  indri::index::DocListIterator::DocumentData* next() {
    indri::index::DocListIterator::DocumentData* entry = NULL;
    indri::index::DocListIterator *iterToMove = NULL;
    for (auto iter: _lists) {
      if (!iter || iter->finished()) {
        continue;
      }

      indri::index::DocListIterator::DocumentData* current = iter->currentEntry();
      if (!entry) {
        entry = current;
        iterToMove = iter;
      } else {
        if (current->document < entry->document) {
          entry = current;
          iterToMove = iter;
        }
      }
    }

    if (iterToMove){
      iterToMove->nextEntry();
    }

    return entry;
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
    // usage( param );

    std::string index = param["index"];
    std::string query = param["query"];
    std::string qno = param.get("qno", "1");
    int count = param.get("count", 1000);

    int k1 = param.get("k1");

    std::string bStringSpec = param["Bf"];
    std::map<std::string, double> fieldB;
    parse_field(fieldB, bStringSpec);

    std::string wtStringSpec = param["Wf"];
    std::map<std::string, double> fieldWt;
    parse_field(fieldWt, wtStringSpec);

    // Use a vector to record all the fields
    std::vector<std::string> fields;
    for (auto f: fieldB) {
      fields.push_back(f.first);
    }

    QueryBM25F bm25f(index, fields, fieldB, fieldWt, k1);
    bm25f.query("1", query, count);
  }
  catch( lemur::api::Exception& e ) {
    LEMUR_ABORT(e);
  } catch( ... ) {
    std::cout << "Caught unhandled exception" << std::endl;
    return -1;
  }

  return 0;
}

static void parse_field( std::map<std::string, double> &m, const std::string& spec ) {
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
}

