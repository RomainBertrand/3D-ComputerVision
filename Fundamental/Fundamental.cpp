// Imagine++ project
// Project:  Fundamental
// Author:   Pascal Monasse

#include "./Imagine/Features.h"
#include <Imagine/Graphics.h>
#include <Imagine/LinAlg.h>
#include <algorithm>
#include <vector>
#include <cstdlib>
#include <ctime>
using namespace Imagine;
using namespace std;

static const float BETA = 0.01f; // Probability of failure

struct Match {
    float x1, y1, x2, y2;
};

// Display SIFT points and fill vector of point correspondences
void algoSIFT(Image<Color,2> I1, Image<Color,2> I2, vector<Match>& matches) {
    // Find interest points
    SIFTDetector D;
    D.setFirstOctave(-1);
    Array<SIFTDetector::Feature> feats1 = D.run(I1);
    drawFeatures(feats1, Coords<2>(0,0));
    cout << "Im1: " << feats1.size() << flush;
    Array<SIFTDetector::Feature> feats2 = D.run(I2);
    drawFeatures(feats2, Coords<2>(I1.width(),0));
    cout << " Im2: " << feats2.size() << flush;

    const double MAX_DISTANCE = 100.0*100.0;
    for(size_t i=0; i < feats1.size(); i++) {
        SIFTDetector::Feature f1=feats1[i];
        for(size_t j=0; j < feats2.size(); j++) {
            double d = squaredDist(f1.desc, feats2[j].desc);
            if(d < MAX_DISTANCE) {
                Match m;
                m.x1 = f1.pos.x();
                m.y1 = f1.pos.y();
                m.x2 = feats2[j].pos.x();
                m.y2 = feats2[j].pos.y();
                matches.push_back(m);
            }
        }
    }
}

float error(FMatrix<float,3,3> FT, float x_1, float y_1, float x_2, float y_2){
    FVector<float,3> x = {x_1, y_1, 1};
    float error = ((FT*x)[0]*x_2 + (FT*x)[1]*y_2 + (FT*x)[2])/sqrt((FT*x)[0]*(FT*x)[0]+(FT*x)[1]*(FT*x)[1]);
    return abs(error);
}

FMatrix<float,3,3> F_from_points(const vector<Match>& matches, const vector<int>& inlierIndexes){
    float f[3][3] = {{0.001, 0, 0}, {0, 0.001, 0}, {0, 0, 1}};
    FMatrix<float,3,3> N(f); // For normalization
    Matrix<float> A(max(9, (int)inlierIndexes.size()), 9); // we don't have to complete manually the last raw
    if (inlierIndexes.size() == 8){
        for (int j=0; j<9; j++){
            A(8, j) = 0;
        }
    }
    for (int j=0; j<inlierIndexes.size();j++){
        FVector<float, 3> x_i = {matches[inlierIndexes[j]].x1, matches[inlierIndexes[j]].y1, 1};
        FVector<float, 3> x_ip = {matches[inlierIndexes[j]].x2, matches[inlierIndexes[j]].y2, 1};
        x_i = N*x_i; x_ip = N*x_ip; // Normalization
        A(j,0) = x_i[0]*x_ip[0];
        A(j,1) = x_i[0]*x_ip[1];
        A(j,2) = x_i[0]*x_ip[2];
        A(j,3) = x_i[1]*x_ip[0];
        A(j,4) = x_i[1]*x_ip[1];
        A(j,5) = x_i[1]*x_ip[2];
        A(j,6) = x_i[2]*x_ip[0];
        A(j,7) = x_i[2]*x_ip[1];
        A(j,8) = x_i[2]*x_ip[2];
    }
    Vector<float> S;                      // Singular value decomposition:
    Matrix<float> U, Vt;
    svd(A,U,S,Vt);
    Vector<float> F = Vt.getRow(8);
    float f_m[3][3] ={{F[0], F[1], F[2]}, {F[3], F[4], F[5]}, {F[6], F[7], F[8]}};
    FMatrix<float,3,3> FM(f_m);
    FVector<float,3> Sf;                      // Singular value decomposition:
    FMatrix<float,3,3> Uf, Vtf;
    svd(FM,Uf,Sf,Vtf);
    Sf[2] = 0; // manual projection
    FMatrix<float,3,3> CurrentF = transpose(N)*Uf*Diagonal(Sf)*Vtf*N; // renormalize

    return CurrentF;
}

// RANSAC algorithm to compute F from point matches (8-point algorithm)
// Parameter matches is filtered to keep only inliers as output.
FMatrix<float,3,3> computeF(vector<Match>& matches) {
    const float distMax = 1.5f; // Pixel error for inlier/outlier discrimination
    int Niter=100000; // Adjusted dynamically
    FMatrix<float,3,3> bestF;
    vector<int> bestInliers;
    // --------------- TODO ------------

    int i = 0;
    while (i<Niter){
        vector<int> inlierIndexes;
        // we pick k=8, as requested in the exercice
        for (int j = 0; j<8; j++){
            int temporary_index = rand() % matches.size();
            inlierIndexes.push_back(temporary_index);
        }
        i+=1;
        FMatrix<float,3,3> CurrentF = F_from_points(matches, inlierIndexes);
        // now we have to compute the errors (determine inliers and outliers)
        vector<int> inliers;
        for (int j=0; j<matches.size(); j++){
            if (error(transpose(CurrentF), matches[j].x1, matches[j].y1, matches[j].x2, matches[j].y2)<distMax){
                inliers.push_back(j);
            }
        }
        if (inliers.size() > bestInliers.size()){
            bestF = CurrentF;
            bestInliers = inliers;
            cout<<bestInliers.size()<<"<- best inliers; total matches ->"<<matches.size()<<endl;
            float denom = log(1-pow(bestInliers.size()/(float)matches.size(), 8));
	    if (denom<0){ // m/n small
		Niter = min(Niter, int(ceil(log(BETA)/denom)));
	    }
            cout<<"Niter "<<Niter<<endl;
        }

    }
    bestF = F_from_points(matches, bestInliers);

    // Updating matches with inliers only
    vector<Match> all=matches;
    matches.clear();
    for(size_t i=0; i<bestInliers.size(); i++)
	matches.push_back(all[bestInliers[i]]);
    return bestF;
}


// Expects clicks in one image and show corresponding line in other image.
// Stop at right-click.
void displayEpipolar(Image<Color> I1, Image<Color> I2, const FMatrix<float,3,3>& F) {
    while(true) {
        int x,y;
        if(getMouse(x,y) == 3)
            break;
        // --------------- TODO ------------
        drawCircle(x, y, 3, RED);
        FVector<int,3> X = {x, y, 1};
	    FVector<float,3> Xp;
        bool inI1 = (x >= I1.width());
        if (inI1){
            X[0] -= I1.width();
            Xp = transpose(F)*X;

        } else {
            Xp = F*X;
        }
        Xp /= Xp[1];
        float y1 = - Xp[2]; // left intersection
        float y2 = - Xp[0]*I1.width() - Xp[2]; // right intersection
        drawLine(I1.width()*inI1, y1, I1.width()+I2.width()*inI1, y2, BLUE);
    }
}

int main(int argc, char* argv[]){
    srand((unsigned int)time(0));

    const char* s1 = argc>1? argv[1]: srcPath("im1.jpg");
    const char* s2 = argc>2? argv[2]: srcPath("im2.jpg");

    // Load and display images
    Image<Color,2> I1, I2;
    if( ! load(I1, s1) || ! load(I2, s2) ) {
        cerr<< "Unable to load images" << endl;
        return 1;
    }
    int w = I1.width();
    openWindow(2*w, I1.height());
    display(I1,0,0);
    display(I2,w,0);

    vector<Match> matches;
    algoSIFT(I1, I2, matches);
    const int n = (int)matches.size();
    cout << " matches: " << n << endl;
    drawString(100,20,std::to_string(n)+ " matches",RED);
    click();

    FMatrix<float,3,3> F = computeF(matches);
    cout << "F="<< endl << F;

    // Redisplay with matches
    display(I1,0,0);
    display(I2,w,0);
    for(size_t i=0; i<matches.size(); i++) {
        Color c(rand()%256,rand()%256,rand()%256);
        fillCircle(matches[i].x1+0, matches[i].y1, 2, c);
        fillCircle(matches[i].x2+w, matches[i].y2, 2, c);
    }
    drawString(100, 20, to_string(matches.size())+"/"+to_string(n)+" inliers", RED);
    click();

    // Redisplay without SIFT points
    display(I1,0,0);
    display(I2,w,0);
    displayEpipolar(I1, I2, F);

    endGraphics();
    return 0;
}
