

/*==========================================================================
 * Copyright (c) 2018 RMIT University.  All Rights Reserved.
 *
 * Use of the Lemur Toolkit for Language Modeling and Information Retrieval
 * is subject to the terms of the software license set forth in the LICENSE
 * file included with this software, and also available at
 * http://www.lemurproject.org/license.html
 *
 *==========================================================================
*/

//
// BM25F
//
// 08 October 2018
//

#include "indri/TokenizerFactory.hpp"
#include "indri/ParserFactory.hpp"
#include "indri/Parameters.hpp"
#include "indri/IndriParser.hpp"
#include "indri/ParsedDocument.hpp"
#include "indri/StemmerFactory.hpp"
#include "indri/NormalizationTransformation.hpp"
#include "indri/UTF8CaseNormalizationTransformation.hpp"

int main(int, char **) {
  try {
    cerr << "Built with " << INDRI_DISTRIBUTION << endl;

    indri::parse::Tokenizer* tokenizer = 0;
    indri::parse::Parser* parser = 0;

    tokenizer = indri::parse::TokenizerFactory::get("word-nomarkup");

    std::vector<std::string> includeTags;
    std::vector<std::string> excludeTags;
    std::vector<std::string> indexTags;
    std::vector<std::string> metadataTags;
    std::map<indri::parse::ConflationPattern *, std::string> conflations;
    parser = indri::parse::ParserFactory::get(
        "text", includeTags, excludeTags, indexTags, metadataTags, conflations);
    indri::api::Parameters empty;
    std::vector<indri::parse::Transformation*> transformations;

    transformations.push_back( new indri::parse::NormalizationTransformation() );
    transformations.push_back( new indri::parse::UTF8CaseNormalizationTransformation() );
    transformations.push_back( indri::parse::StemmerFactory::get("krovetz", empty) );

    std::string line;
    while(std::getline(std::cin, line)){
      indri::parse::UnparsedDocument document = {.text = line.c_str(), .textLength = line.length(), .content=line.c_str(), .contentLength = line.length()};
      indri::parse::TokenizedDocument* tokenized = tokenizer->tokenize(&document);
      indri::api::ParsedDocument *parsed = parser->parse( tokenized );
      for (size_t i = 0; i < transformations.size(); i++) {
        parsed = transformations[i]->transform(parsed);
      }

      // std::cout << parsed->positions.size() << " " << parsed->terms.size() << std::endl;
      // std::cout << parsed->text << std::endl;
      // std::cout << parsed->content << std::endl;
      for (size_t i = 0; i < parsed->terms.size(); ++i) {
        if (i < parsed->terms.size() - 1) {
          std::cout << parsed->terms[i] << " ";
        } else {
          std::cout << parsed->terms[i] << std::endl;
        }
      }
    }
  }
  catch( lemur::api::Exception& e ) {
    LEMUR_ABORT(e);
  } catch( ... ) {
    std::cout << "Caught unhandled exception" << std::endl;
    return -1;
  }

  return 0;
}
