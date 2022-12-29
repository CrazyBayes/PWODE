#ifndef PWODE_H
#define PWODE_H


#pragma once

#include "xxyDist.h"
#include "incrementalLearner.h"
#include "crosstab.h"

class PWODE : public IncrementalLearner
{ public:
        PWODE(char* const *& argv, char* const * end);
        virtual ~PWODE();

        void getCapabilities(capabilities &c);
        bool trainingIsFinished();
        void reset(InstanceStream &is);
        void initialisePass();
        void train(const instance &inst);
        void finalisePass();
        void classify(const instance &inst, std::vector<double> &posteriorDist);

    protected:
        void subclassify(const instance &inst, CategoricalAttribute sp, std::vector<double> &classDist);
        double getLocalWeight(const instance &inst, CategoricalAttribute sp);
        float getMidValue(CategoricalAttribute x1, CategoricalAttribute x2, CatValue yT);
        float getMidValueYyyy(CategoricalAttribute x1, CatValue yT);
    private:
        InstanceStream *instanceStream;
        bool trainingIsFinished_;
        xxyDist xxyDist_;
        unsigned int noCatAtts_;
        unsigned int noClasses_;
        std::vector<float> threshold;
        std::vector<float> yThreshold;
};


#endif // PWODE_H
