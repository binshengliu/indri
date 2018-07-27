

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
#include "indri/TrecRunFile.hpp"

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

static void printGrams( const std::string& query, const std::vector<indri::query::RelevanceModel::Gram*>& grams ) {
  std::cout << "# query: " << query << std::endl;
  for( size_t j=0; j<grams.size(); j++ ) {
    std::cout << std::setw(15)
              << std::setprecision(15)
              << std::fixed
              << grams[j]->weight << " ";
    std::cout << grams[j]->terms.size() << " ";

    for( size_t k=0; k<grams[j]->terms.size(); k++ ) {
      std::cout << grams[j]->terms[k] << " ";
    }

    std::cout << std::endl;
  }
}

static void printQuery(const std::string& query, const std::string &fieldName, int documents,
                        const std::vector<indri::query::RelevanceModel::Gram*>& grams) {
  std::cout << "  <model query=\"" << query
            << "\" documents=\"" << documents
            << "\" field=\"" << fieldName
            << "\">" << std::endl;
  for( size_t j=0; j<grams.size(); j++ ) {
    std::cout << "    ";
    std::cout << std::setw(15)
              << std::setprecision(15)
              << std::fixed
              << grams[j]->weight << " ";

    for( size_t k=0; k<grams[j]->terms.size(); k++ ) {
      std::cout << grams[j]->terms[k] << " ";
    }
    std::cout << std::endl;
  }

  std::cout << "  </model>" << std::endl;
  std::cout << std::endl;
}

static void usage(indri::api::Parameters param) {
  if( !param.exists( "trecrun" ) || !( param.exists( "index" ) || param.exists( "server" ) ) || !param.exists( "documents" )
      || !param.exists("field")) {
   std::cerr << "rmodel usage: " << std::endl
             << "   rmodel -field=myfield -trecrun=myrun -index=myindex -smoothing=method:lbs,mu:2500,beta:0.3 -documents=10 -maxGrams=2 -terms=50 -format=xml" << std::endl
             << "     myfield: a valid field in the index, or \"all\" for whole document" << std::endl
             << "     myrun: a valid Indri run file (be sure to use quotes around it if there are spaces in it)" << std::endl
             << "     myindex: a valid Indri index" << std::endl
             << "     documents: the number of documents to use to build the relevance model" << std::endl
             << "     maxGrams (optional): maximum length (in words) of phrases to be added to the model, default is 1 (unigram)" << std::endl
             << "     terms (optional): the number of terms of the final rm" << std::endl
             << "     format (optional): write into xml format" << std::endl
       ;
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
    indri::api::Parameters& param = indri::api::Parameters::instance();
    param.loadCommandLine( argc, argv );
    usage( param );

    std::string trecrun = param["trecrun"];
    std::string rmSmoothing = param.get("smoothing", ""); // eventually, we should offer relevance model smoothing
    std::string field = param["field"];
    int documents = (int) param[ "documents" ];
    int maxGrams = (int) param.get( "maxGrams", 1 ); // unigram is default
    int terms = (int)param.get("terms", 0);
    bool xmlFormat = (param.get("format", "") == "xml");

    std::ifstream ifs(trecrun);
    if (!ifs) {
      std::cerr << "Open " << trecrun << " failed." << std::endl;
      return EXIT_FAILURE;
    }

    indri::api::QueryEnvironment environment;
    open_indexes( environment, param );

    indri::query::TrecRunFile trec;
    std::vector<indri::query::TrecQueryResult> results = trec.load(ifs, documents);

    if (xmlFormat) {
      std::cout << "<root>" << std::endl;
      std::cout << "  <run>" << trecrun << "</run>" << std::endl << std::endl;
    }

    for (size_t query_index = 0; query_index < results.size(); ++query_index) {
      std::cerr << "\r Processed: "
                << query_index + 1
                << "/"
                << results.size() << std::flush;
      std::vector<indri::query::TrecRecord> records = results[query_index].records;

      indri::query::RelevanceModel model( environment, rmSmoothing, maxGrams, documents );
      model.generate(records, field);
      if (terms != 0) {
        model.normalize(terms);
      }

      const std::vector<indri::query::RelevanceModel::Gram*>& grams = model.getGrams();
      if (xmlFormat) {
        printQuery(results[query_index].queryNumber, field, documents, grams);
      } else {
        printGrams( results[query_index].queryNumber, grams );
      }
    }

    if (xmlFormat) {
      std::cout << "</root>" << std::endl;
    }
  } catch( lemur::api::Exception& e ) {
    LEMUR_ABORT(e);
  } catch( ... ) {
    std::cout << "Caught an unhandled exception" << std::endl;
  }

  return 0;
}

