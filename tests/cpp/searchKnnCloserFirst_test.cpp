// This is a test file for testing the interface
//  >>> virtual std::vector<std::pair<dist_t, labeltype>>
//  >>>    searchKnnCloserFirst(const void* query_data, size_t k) const;
// of class AlgorithmInterface

#include "hnswlib.h"
#include <assert.h>
#include <vector>
#include <iostream>

namespace {

using idx_t = hnswlib::labeltype;

void test() {
//    int d = 4;
//    idx_t n = 100;
//    idx_t nq = 10;
//    size_t k = 10;
    int d = 2;
    idx_t n = 5;
    idx_t nq = 2;
    size_t k = 1;

    std::vector<float> data(n * d);
    std::vector<float> query(nq * d);

    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib;

    for (int i = 0; i < n * d; ++i) {
        data[i] = (float)i+2;
    }

    query[0] = 43;
    query[1] = 44;
    query[2] = 143;
    query[2] = 144;

    /*for (idx_t i = 0; i < n * d; ++i) {
        data[i] = distrib(rng);
        std::cout<<"input: "<<data[i]<<std::endl;
    }
    for (idx_t i = 0; i < nq * d; ++i) {
        query[i] = distrib(rng);
        std::cout<<"query: "<<query[i]<<std::endl;
    }*/

    hnswlib::L2Space space(d);
    hnswlib::AlgorithmInterface<float>* alg_brute  = new hnswlib::BruteforceSearch<float>(&space, 2 * n);
    hnswlib::AlgorithmInterface<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, 2 * n);

    for (size_t i = 0; i < n; ++i) {
        alg_brute->addPoint(data.data() + d * i, i);
        alg_hnsw->addPoint(data.data() + d * i, i);
    }

    cout<<"done building graph!!!"<<endl;
    getchar();

    // test searchKnnCloserFirst of BruteforceSearch
    /*for (size_t j = 0; j < nq; ++j) {
        const void* p = query.data() + j * d;
        auto gd = alg_brute->searchKnn(p, k);
        auto res = alg_brute->searchKnnCloserFirst(p, k);
        assert(gd.size() == res.size());
        size_t t = gd.size();
        while (!gd.empty()) {
            assert(gd.top() == res[--t]);
            gd.pop();
        }
    }*/
    for (size_t j = 0; j < nq; ++j) {
        const void* p = query.data() + j * d;
        auto gd = alg_hnsw->searchKnn(p, k);
        auto res = alg_hnsw->searchKnnCloserFirst(p, k);

        for(auto r:res) std::cout<<"dist: "<<r.first<<" result: "<<r.second<<std::endl;
        assert(gd.size() == res.size());
        size_t t = gd.size();
        while (!gd.empty()) {
            assert(gd.top() == res[--t]);
            gd.pop();
        }
    }

    delete alg_brute;
    delete alg_hnsw;
}
}  // namespace

int main0() {

    std::cout << "Testing ..." << std::endl;
    test();
    std::cout << "Test ok" << std::endl;

    return 0;
}


class skiplist
{
private:
    static constexpr int MAX_LEVEL = 2;
    static constexpr double P = 0.6;
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<> dis;

    struct node
    {
        int data{-1};
        node *next[MAX_LEVEL+1]{};

        node()
        {
            for(int i=1;i<=MAX_LEVEL;i++)
                next[i] = nullptr;
        }
    };

    node *head;

    int randomLevel()
    {
        int ret = 1;
        while(ret<MAX_LEVEL && dis(gen)<P)
            ret++;
        return ret;
    }


public:
    skiplist()
    {
        head = new node;
        gen = std::mt19937(rd());
        dis = std::uniform_real_distribution<> (0,1);
    }

    void insert(int x)
    {
        node *tmp = new node;
        tmp->data = x;
        node *pre[MAX_LEVEL+1];
        node *now = head;
        int level = randomLevel();
        cout<<"level: "<<level<<endl;
        for(int i=MAX_LEVEL;i>=1;i--)
        {
            while(now->next[i] != NULL && now->next[i]->data<x) now = now->next[i];
            pre[i] = now;
        }

        for(int i=level;i>=1;i--)
        {
            tmp->next[i] = pre[i]->next[i];
            pre[i]->next[i] = tmp;
        }
    }

    void print()
    {

        node *now = head->next[2];
        while(now != NULL)
        {
            printf("level0: %d ",now->data);
            now = now->next[2];
        }
        printf("\n");

        node *c= head->next[1];
        while(c != NULL)
        {
            printf("level0: %d ",c->data);
            c = c->next[1];
        }
        printf("\n");
    }
};

#include "opencv2/opencv.hpp"
int main(){
    vector<cv::Point2f>vec;
    vec.emplace_back(1,2);
    vec.emplace_back(3,4);
    cv::Mat xpts(1,2,CV_32F,&vec[0].x,2*sizeof(float));
    cout<<xpts.step<<endl;
    cout<<xpts.step1()<<endl;
    cout<<xpts<<endl;

    cv::AutoBuffer<float> dbuf(10);
    cv::Mat img(1,10,CV_32F,dbuf);
    for (int i = 0; i <10 ; ++i) {
        cout<<dbuf[i]<<" ";
    }

    img = 17;
    cout<<endl;
    for (int i = 0; i <10 ; ++i) {
        cout<<dbuf[i]<<" ";
    }


//    skiplist L;
//    for(int i=6;i>=1;i--)
//        L.insert(i);
//    L.print();
}
