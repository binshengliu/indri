

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

    indri::collection::Repository r;
    r.openRead(param["index"]);

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
    std::set<std::string> fieldSet(fields.begin(), fields.end());

    indri::collection::Repository::index_state state = r.indexes();
    indri::index::Index* index = (*state)[0];

    double totalDocumentCount = index->documentCount();

    // Average field length;
    std::map<std::string, double> avgFieldLen;
    for (auto f: fields) {
      double len = index->fieldTermCount(f) / totalDocumentCount;
      avgFieldLen[f] = len;
    }

    std::string term = "okapi";
    std::string stem = r.processTerm(term);

    std::vector<std::string> stems;
    std::vector<indri::index::DocListIterator*> listIters;

    indri::thread::ScopedLock( index->iteratorLock() );

    for (auto s: stems) {
      listIters.push_back(index->docListIterator(s));
    }

    indri::index::DocListIterator* iter = index->docListIterator( stem );
    if (iter == NULL) return 0;

    iter->startIteration();

    double termDocCount = index->documentCount(term);
    indri::index::DocListIterator::DocumentData* entry;
    std::map<lemur::api::DOCID_T, double> docScores;
    int i = 0;
    for( iter->startIteration(); iter->finished() == false; iter->nextEntry() ) {
      i += 1;
      entry = (indri::index::DocListIterator::DocumentData*) iter->currentEntry();
      std::cerr << entry->document << " " << i << "/" << termDocCount << std::endl;

      const indri::index::TermList* termList = index->termList( entry->document );
      indri::api::DocumentVector* result = new indri::api::DocumentVector( index, termList);
      std::vector<indri::api::DocumentVector::Field>& fields = result->fields();
      std::vector<std::string>& stems = result->stems();
      const std::vector<int>& positions = result->positions();
      std::map<std::string, int> fOcc;
      std::map<std::string, int> docFieldLen;
      for (size_t f = 0; f < fields.size(); f++) {
        std::string fn = fields[f].name;
        // Not the field we want.
        if (fieldSet.find(fn) == fieldSet.end()) {
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
          if (stems[positions[pos]] == stem) {
            if (fOcc.find(fields[f].name) == fOcc.end()) {
              fOcc[fields[f].name] = 0;
            }
            fOcc[fields[f].name] += 1;
          }
        }
        // std::cout << term << " " << fields[f].name << " " << count << std::endl;
      }

      double pseudoFreq = 0;
      for (auto it: fOcc) {
        std::string f = it.first;
        int occ = it.second;

        double fieldFreq = occ / (1 + fieldB[f] * (docFieldLen[f] / avgFieldLen[f] - 1));
        pseudoFreq += fieldWt[f] * fieldFreq;
      }

      double tf = pseudoFreq / (k1 + pseudoFreq);

      double idf = (totalDocumentCount - termDocCount + 0.5) / (termDocCount + 0.5);

      docScores[0] += tf * idf;
    }

    delete iter;
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

