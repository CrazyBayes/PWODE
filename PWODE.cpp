#include "PWODE.h"
#include "utils.h"
#include <assert.h>
#include "correlationMeasures.h"
#include <algorithm>
#include "globals.h"

PWODE::PWODE(char* const *& argv, char* const * end):
    xxyDist_()
{
    name_ = "PWODE";


    trainingIsFinished_ = false;
}

PWODE::~PWODE()
{
}

void PWODE::getCapabilities(capabilities &c)
{
    c.setCatAtts(true);
}
bool PWODE::trainingIsFinished()
{
    return trainingIsFinished_;
}
void PWODE::reset(InstanceStream &is)
{

    instanceStream = &is;
    noCatAtts_ = is.getNoCatAtts();
    noClasses_ = is.getNoClasses();
    xxyDist_.reset(is);
    trainingIsFinished_ = false;
    threshold.resize(noClasses_);
    yThreshold.resize(noClasses_);

}

void PWODE::initialisePass()
{
    assert(trainingIsFinished_ == false);
}
void PWODE::train(const instance &inst)
{
    assert(trainingIsFinished_ == false);
    xxyDist_.update(inst);
}

float PWODE::getMidValue(CategoricalAttribute x1, CategoricalAttribute x2, CatValue yT)
{
    //printf("getMidValue start\n");

    std::vector<float> lcmiXiXj;
    lcmiXiXj.clear();
    float m = 0.0;
    const double totalCount = xxyDist_.xyCounts.count;
    for (CatValue v1 = 0; v1 < xxyDist_.getNoValues(x1); v1++)
    {
        for (CatValue v2 = 0; v2 < xxyDist_.getNoValues(x2); v2++)
        {

            //for (CatValue y = 0; y < xxyDist_.getNoClasses(); y++)
            //{
            const double x1x2y = xxyDist_.getCount(x1, v1, x2, v2, yT);
            if (x1x2y)
            {

                m = (x1x2y / totalCount) * log2(xxyDist_.xyCounts.getClassCount(yT) * x1x2y /
                                                (static_cast<double> (xxyDist_.xyCounts.getCount(x1, v1, yT)) *
                                                 xxyDist_.xyCounts.getCount(x2, v2, yT)));


                lcmiXiXj.push_back(m);
            }
            // }

        }
    }
    float midValue = 0.0;
    if(lcmiXiXj.size() > 0)
    {
        std::sort(lcmiXiXj.begin(),lcmiXiXj.end());
        int lengthOflcmi = lcmiXiXj.size();

        if(lengthOflcmi%2==0)
        {
            midValue = ((lcmiXiXj[lengthOflcmi/2-1]+lcmiXiXj[lengthOflcmi/2])/2);
        }
        else
        {
            midValue = lcmiXiXj[lengthOflcmi/2];
        }

    }

    //printf("getMidValue end\n");
    return midValue;

}
float PWODE::getMidValueYyyy(CategoricalAttribute x1, CatValue yT)
{
    //printf("getMidValueYyyy start\n");
    float m = 0.0;
    std::vector<float> mixiy;
    mixiy.clear();
    const double totalCount = xxyDist_.xyCounts.count;
    for (CatValue v = 0; v < xxyDist_.getNoValues(x1); v++)
    {
        //for (CatValue y = 0; y < xxyDist_.getNoClasses(); y++)
        //{
        const InstanceCount avyCount = xxyDist_.xyCounts.getCount(x1, v, yT);

        if (avyCount)
        {
            m = (avyCount / totalCount) * log2(avyCount / ((xxyDist_.xyCounts.getCount(x1, v) / totalCount)
                                               * xxyDist_.xyCounts.getClassCount(yT)));
            mixiy.push_back(m);

        }
        //}
    }
    //printf("getMidValueYyyy mid\n");
    float midValue = 0.0;
    if(mixiy.size()> 0 )
    {
        std::sort(mixiy.begin(),mixiy.end());
        int lengthOflmi = mixiy.size();

        if(lengthOflmi%2==0)
        {
            midValue = ((mixiy[lengthOflmi/2-1]+mixiy[lengthOflmi/2])/2);
        }
        else
        {
            midValue = mixiy[lengthOflmi/2];
        }
    }
    //printf("getMidValueYyyy end\n");
    return midValue;

}

void PWODE::finalisePass()
{

    assert(trainingIsFinished_ == false);

    std::vector<float> midValueListyyy;
    float sumyyy=0.0;
    for(CatValue yT = 0; yT < noClasses_; yT++)
    {
        midValueListyyy.clear();
        sumyyy = 0.0;
        for(CategoricalAttribute xx = 0; xx < noCatAtts_; xx++)
        {
            float myMidValue = getMidValueYyyy(xx,yT);
            midValueListyyy.push_back(myMidValue);
            sumyyy+= myMidValue;
        }
        float yyythresholdCandidte = sumyyy/(1.0*midValueListyyy.size());
        yThreshold[yT] = yyythresholdCandidte;
        //消融实验
        //yThreshold[yT] = -10000.0;
    }



    std::vector<float> midValueList;
    float sum = 0.0;

    for(CatValue yT = 0; yT < noClasses_; yT++)
    {
        midValueList.clear();
        sum = 0.0;
        for(CategoricalAttribute x1 = 0; x1 < noCatAtts_; x1++)
        {
            for(CategoricalAttribute x2 = 0; x2 < x1; x2++)
            {

                float cuMidValue = getMidValue(x1,x2,yT);
                midValueList.push_back(cuMidValue);
                sum += cuMidValue;

            }
        }
        float thresholdByCandidate = sum/(1.0*midValueList.size());
        threshold[yT] = thresholdByCandidate;
       // threshold[yT] = -10000.0;
    }


    trainingIsFinished_ = true;
    // printf("finalisePass\n");

}
void PWODE::subclassify(const instance &inst, CategoricalAttribute sp, std::vector<double> &classDist)
{

    for(CatValue y = 0; y < noClasses_; y++)
    {
        classDist[y] = xxyDist_.xyCounts.jointP(sp, inst.getCatVal(sp),y);
    }
    for(CategoricalAttribute x = 0; x < noCatAtts_; x++)
    {
        if( x!=sp )
        {
            for(CatValue y = 0; y < noClasses_; y++ )
            {

                classDist[y] *= xxyDist_.p(x, inst.getCatVal(x), sp, inst.getCatVal(sp),y);

            }
        }
    }

}


double PWODE::getLocalWeight(const instance &inst, CategoricalAttribute sp)
{

    double weight = 0.0;
    double total = xxyDist_.xyCounts.count;

    /*
    if(mi_loc[sp] < yThreshold*noClasses_)
    {
        for(CatValue y = 0; y < noClasses_; y++)
        {
            double xy = xxyDist_.xyCounts.getCount(sp, inst.getCatVal(sp), y);
            double xCount = xxyDist_.xyCounts.getCount(sp, inst.getCatVal(sp));
            double yCount = xxyDist_.xyCounts.getClassCount(y);
            if(xy > 0)
            {
                weight += log2((xCount/total)*(yCount/total));
            }
        }
    }
    else
    {
        for(CatValue y = 0; y < noClasses_; y++)
        {
            double xy = xxyDist_.xyCounts.getCount(sp, inst.getCatVal(sp), y);
            if(xy > 0)
            {
                weight += log2((xy/total));
            }
        }
    }

    for(CategoricalAttribute x = 0 ; x  < noCatAtts_; x++)
    {
        if(x != sp )
        {

            if(cmi_loc[sp][x] < threshold*noClasses_)
            {

                for(CatValue y = 0; y < noClasses_; y++)
                {

                    double xiy = xxyDist_.xyCounts.getCount(x,inst.getCatVal(x),y);
                    if(xiy > 0)
                    {
                        double yLabel = xxyDist_.xyCounts.getClassCount(y);
                        weight += log2(xiy/yLabel);
                    }
                }
            }
            else
            {
                for(CatValue y = 0; y < noClasses_; y++)
                {
                    double xxy = xxyDist_.getCount(sp, inst.getCatVal(sp), x, inst.getCatVal(x),y);
                    if(xxy > 0)
                    {
                        double xy = xxyDist_.xyCounts.getCount(sp, inst.getCatVal(sp),y);
                        weight += log2((xxy/xy));
                    }
                }
            }
        }
    }*/
    float mixy = 0.0;
    for(CatValue y =0; y < noClasses_; y++)
    {
        mixy = 0.0;
        const InstanceCount avyCount = xxyDist_.xyCounts.getCount(sp,inst.getCatVal(sp), y);
        if (avyCount)
        {
            mixy= (avyCount / total) * log2(avyCount / ((xxyDist_.xyCounts.getCount(sp, inst.getCatVal(sp)) / total)
                                            * xxyDist_.xyCounts.getClassCount(y)));
            if(mixy < yThreshold[y])
            {
                indepependenceRemove += 1;
                weight += log2(xxyDist_.xyCounts.p(y)*xxyDist_.xyCounts.p(sp,inst.getCatVal(sp)));
            }
            else
            {
                weight += log2(xxyDist_.xyCounts.jointP(sp,inst.getCatVal(sp),y));
            }
        }
    }
    float cmixxy = 0.0;
    for(CategoricalAttribute x = 0; x < noCatAtts_; x++)
    {
        cmixxy = 0.0;
        for(CatValue y = 0; y < noClasses_; y++)
        {
            if(x!=sp)
            {
                const double x1x2y = xxyDist_.getCount(sp, inst.getCatVal(sp), x, inst.getCatVal(x), y);
                if (x1x2y)
                {
                    cmixxy= (x1x2y / total) * log2(xxyDist_.xyCounts.getClassCount(y) * x1x2y /
                                                   (static_cast<double> (xxyDist_.xyCounts.getCount(sp, inst.getCatVal(sp), y)) *
                                                    xxyDist_.xyCounts.getCount(x, inst.getCatVal(x), y)));
                    if(cmixxy < threshold[y])
                    {
                        indepependenceRemove += 1;
                        weight += log2(xxyDist_.xyCounts.p(x, inst.getCatVal(x),y));
                    }
                    else
                    {
                        weight += log2(xxyDist_.p(x,inst.getCatVal(x), sp, inst.getCatVal(sp), y));
                    }
                }
            }
        }
    }
    return weight;
}

void PWODE::classify(const instance &inst, std::vector<double> &posteriorDist)
{
    //crosstab<float> cmi_loc = crosstab<float>(noCatAtts_);
    //getCondMutualInfloc(xxyDist_, cmi_loc, inst);
    //std::vector<float> mi_loc;
    //getMutualInformationloc(xxyDist_.xyCounts, mi_loc, inst);
    indepependenceRemove = 0;

    std::vector<double> localWeight;
    localWeight.clear();
    localWeight.resize(noCatAtts_);
    float localMinValue = std::numeric_limits<float>::max();
    float localMaxValue = -std::numeric_limits<float>::max();


    for(CategoricalAttribute lsp = 0; lsp < noCatAtts_; lsp++)
    {
        localWeight[lsp] = getLocalWeight(inst, lsp);

        if(localMaxValue < localWeight[lsp])
            localMaxValue  = localWeight[lsp];
        if(localMinValue > localWeight[lsp])
            localMinValue = localWeight[lsp];
    }


    float weightSum = 0;
    if(fabs(localMaxValue - localMinValue)<0.000001)
    {
        for(CategoricalAttribute lsp = 0; lsp < noCatAtts_; lsp++)
        {
            localWeight[lsp] = 1;
            weightSum += localWeight[lsp];
        }
    }
    else
    {
        for(CategoricalAttribute lsp = 0; lsp < noCatAtts_; lsp++)
        {
            localWeight[lsp] = (localWeight[lsp] - localMinValue);

            weightSum += localWeight[lsp];
        }
    }

    for(int i = 0; i < localWeight.size(); i++ )
    {
        localWeight[i] = localWeight[i]/weightSum;

    }
    independenceRemoveRatePerInstance += (1.0*indepependenceRemove)/(1.0*noClasses_*
                    noCatAtts_*(2*noCatAtts_-1));

    //printf("%d\n",indepependenceRemove);

    for(CatValue y = 0; y < noClasses_; y++)
    {
        posteriorDist[y] = 0.0;
    }
    /*

    for(CategoricalAttribute x = 0; x < noCatAtts_; x++)
    {
        localWeight[x] = 1.0/noCatAtts_;
    }*/
    std::vector<double> classDist;
    for(CategoricalAttribute sp = 0; sp < noCatAtts_; sp++)
    {
        classDist.clear();
        classDist.resize(noClasses_);
        if(xxyDist_.xyCounts.getCount(sp, inst.getCatVal(sp)) > 0)
        {
            subclassify(inst, sp, classDist);
            for(CatValue y = 0; y < noClasses_; y++)
            {
                posteriorDist[y] += localWeight[sp] * classDist[y];
            }
        }
    }
    normalise(posteriorDist);
}
