// Imagine++ project
// Project:  Panorama
// Author:   Pascal Monasse
// Date:     2013/10/08

#include <Imagine/Graphics.h>
#include <Imagine/Images.h>
#include <Imagine/LinAlg.h>
#include <vector>
#include <sstream>
using namespace Imagine;
using namespace std;

void drawPoint(IntPoint2 point){
    fillCircle(point, 1, RED);
}

// Record clicks in two images, until right button click
void getClicks(Window w1, Window w2,vector<IntPoint2>& pts1, vector<IntPoint2>& pts2) {
    int button = 0;
    IntPoint2 p;
    while (true){
        if (button == 3 and min(pts1.size(), pts2.size())>=4){
            break;
        }
        if (pts2.size() < pts1.size()){
            setActiveWindow(w2);
            button = getMouse(p);
            pts2.push_back(p);
            drawPoint(p);
        } else {
            setActiveWindow(w1);
            button = getMouse(p);
            pts1.push_back(p);
            drawPoint(p);
        }
    }
}

// Return homography compatible with point matches
Matrix<float> getHomography(const vector<IntPoint2>& pts1, const vector<IntPoint2>& pts2) {
    size_t n = min(pts1.size(), pts2.size());
    if(n<4) {
        cout << "Not enough correspondences: " << n << endl;
        return Matrix<float>::Identity(3);
    }
    Matrix<double> A(2*n,8);
    Vector<double> B(2*n);
    // ------------- TODO/A completer ----------
    for (size_t i=0; i<n; i++){
        A(2*i, 0) = pts1[i].x();
        A(2*i, 1) = pts1[i].y();
        A(2*i, 2) = 1;
        A(2*i, 3) = 0;
        A(2*i, 4) = 0;
        A(2*i, 5) = 0;
        A(2*i, 6) = -pts2[i].x()*pts1[i].x();
        A(2*i, 7) = -pts2[i].x()*pts1[i].y();
        B[2*i] = pts2[i].x();

        A(2*i+1, 0) = 0;
        A(2*i+1, 1) = 0;
        A(2*i+1, 2) = 0;
        A(2*i+1, 3) = pts1[i].x();
        A(2*i+1, 4) = pts1[i].y();
        A(2*i+1, 5) = 1;
        A(2*i+1, 6) = -pts2[i].y()*pts1[i].x();
        A(2*i+1, 7) = -pts2[i].y()*pts1[i].y();
        B[2*i+1] = pts2[i].y();
    }

    B = linSolve(A, B);
    Matrix<float> H(3, 3);
    H(0,0)=B[0]; H(0,1)=B[1]; H(0,2)=B[2];
    H(1,0)=B[3]; H(1,1)=B[4]; H(1,2)=B[5];
    H(2,0)=B[6]; H(2,1)=B[7]; H(2,2)=1;

    // Sanity check
    cout<<"Sanity check : should be close to 0"<<endl;
    for(size_t i=0; i<n; i++) {
        float v1[]={(float)pts1[i].x(), (float)pts1[i].y(), 1.0f};
        float v2[]={(float)pts2[i].x(), (float)pts2[i].y(), 1.0f};
        Vector<float> x1(v1,3);
        Vector<float> x2(v2,3);
        x1 = H*x1;
        cout << x1[1]*x2[2]-x1[2]*x2[1] << ' '
        << x1[2]*x2[0]-x1[0]*x2[2] << ' '
        << x1[0]*x2[1]-x1[1]*x2[0] << endl;
    }
    cout<<"Sanity check ended"<<endl;
    return H;
}

// Grow rectangle of corners (x0,y0) and (x1,y1) to include (x,y)
void growTo(float& x0, float& y0, float& x1, float& y1, float x, float y) {
    if(x<x0) x0=x;
    if(x>x1) x1=x;
    if(y<y0) y0=y;
    if(y>y1) y1=y;
}

// Panorama construction
void panorama(const Image<Color,2>& I1, const Image<Color,2>& I2, Matrix<float> H) {
    Vector<float> v(3);
    float x0=0, y0=0, x1=I2.width(), y1=I2.height();

    v[0]=0; v[1]=0; v[2]=1;
    v=H*v; v/=v[2];
    growTo(x0, y0, x1, y1, v[0], v[1]);

    v[0]=I1.width(); v[1]=0; v[2]=1;
    v=H*v; v/=v[2];
    growTo(x0, y0, x1, y1, v[0], v[1]);

    v[0]=I1.width(); v[1]=I1.height(); v[2]=1;
    v=H*v; v/=v[2];
    growTo(x0, y0, x1, y1, v[0], v[1]);

    v[0]=0; v[1]=I1.height(); v[2]=1;
    v=H*v; v/=v[2];
    growTo(x0, y0, x1, y1, v[0], v[1]);

    cout << "x0 x1 y0 y1=" << x0 << ' ' << x1 << ' ' << y0 << ' ' << y1<<endl;

    Image<Color> I(int(x1-x0), int(y1-y0));
    setActiveWindow( openWindow(I.width(), I.height()) );
    I.fill(WHITE);


    // push
    /*
    for (int i=0; i<I1.height(); i++){
        for (int j=0; j<I1.width(); j++){
            v[0] = j - x0 ; v[1] = i - y0; v[2] = 1;
            v = H*v; v/=v[2];
            if (min(v[0], v[1])>=0 and v[0]<I.width() and v[1]<I.height()){
                I(v[0],v[1]) = I1(j, i);
            }
        }
    }
    for (int i=0; i<I2.height(); i++){
        for (int j=0; j<I2.width(); j++){
            v[0] = j - x0; v[1]=i - y0; v[2] = 1;
            v = H*v; v/=v[2];
            v[0] -= x0;
            v[1] -= y0;
            if (min(v[0], v[1])>=0 and v[0]<I.width() and v[1]<I.height()){
                Color I2pixel = I2(j, i);
                if (I(v[0], v[1]) != WHITE){
                    I(v[0], v[1]).r() = 0.5 * (I(v[0], v[1]).r() + I2pixel.r());
                    I(v[0], v[1]).g() = 0.5 * (I(v[0], v[1]).g() + I2pixel.g());
                    I(v[0], v[1]).b() = 0.5 * (I(v[0], v[1]).b() + I2pixel.b());
                } else {
                    I(v[0],v[1]) = I2pixel;
                }
            }
        }
    }*/


    // pull
    Matrix<float> Hinv = inverse(H);
    for (int i=0; i<I.height();i++){
        for (int j=0; j<I.width();j++){
            v[0] = j + x0; v[1]=i+y0; v[2]=1;
            if (min(v[0], v[1])>=0 and v[0]<I2.width() and v[1]<I2.height()){
                I(j, i) = I2.interpolate(v[0], v[1]);
            }
            v = Hinv*v; v/=v[2];
            if (min(v[0], v[1])>=0 and v[0]<I1.width() and v[1]<I1.height()){
                Color I1pixel = I1.interpolate(v[0], v[1]);
                if (I(j, i) != WHITE){
                    I(j,i).r() = 0.5 * (I(j,i).r() + I1pixel.r());
                    I(j,i).g() = 0.5 * (I(j,i).g() + I1pixel.g());
                    I(j,i).b() = 0.5 * (I(j,i).b() + I1pixel.b());
                } else {
                    I(j,i) = I1pixel;
                }
            }
        }
    }
    display(I,0,0);
}

// Main function
int main(int argc, char* argv[]) {
    const char* s1 = argc>1? argv[1]: srcPath("image0006.jpg");
    const char* s2 = argc>2? argv[2]: srcPath("image0007.jpg");

    // Load and display images
    Image<Color> I1, I2;
    if( ! load(I1, s1) ||
    ! load(I2, s2) ) {
        cerr<< "Unable to load the images" << endl;
        return 1;
    }
    Window w1 = openWindow(I1.width(), I1.height(), s1);
    display(I1,0,0);
    Window w2 = openWindow(I2.width(), I2.height(), s2);
    setActiveWindow(w2);
    display(I2,0,0);

    // Get user's clicks in images
    vector<IntPoint2> pts1, pts2;
    //getClicks(w1, w2, pts1, pts2);
    // if lazy, only use this pre-recorded clicks
    pts1.push_back(IntPoint2(517,128));
    pts1.push_back(IntPoint2(748,109));
    pts1.push_back(IntPoint2(482,447));
    pts1.push_back(IntPoint2(712,415));

    pts2.push_back(IntPoint2(62,124));
    pts2.push_back(IntPoint2(293,107));
    pts2.push_back(IntPoint2(25,443));
    pts2.push_back(IntPoint2(254,411));

    vector<IntPoint2>::const_iterator it;
    cout << "pts1="<<endl;
    for(it=pts1.begin(); it != pts1.end(); it++)
    cout << *it << endl;
    cout << "pts2="<<endl;
    for(it=pts2.begin(); it != pts2.end(); it++)
    cout << *it << endl;

    // Compute homography
    Matrix<float> H = getHomography(pts1, pts2);
    cout << "H=" << H/H(2,2);

    // Apply homography
    panorama(I1, I2, H);

    endGraphics();
    return 0;
}
