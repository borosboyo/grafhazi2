//=============================================================================================
// 
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Boros Gergo
// Neptun : IGMEF9
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

const char* const vertexSource = R"(
	#version 330
	precision highp float;
	
	uniform vec3 wLookAt, wRight, wUp;
	layout(location = 0) in vec2 cCamWindowVertex;
	out vec3 p;

	void main(){
		gl_Position = vec4(cCamWindowVertex, 0, 1);
		p = wLookAt + wRight * cCamWindowVertex.x + wUp * cCamWindowVertex.y;
	}
)";

const char* fragmentSource = R"(
	#version 330
	precision highp float;

	const vec3 La = vec3(0.5f,0.6f,0.7f);
	const vec3 Le = vec3(0.8f,0.8f,0.8f);
	const vec3 lightPosition = vec3(0.4f,0.4f,0.25f);
	const vec3 ka = vec3(0.5f,0.5f,0.5f);
	const float shininess = 500.0f;
	const int maxdepth = 5;
	const float epsilon = 0.01f;

	struct Hit {
		float t;
		vec3 position, normal;
		int mat;
	};

	struct Ray {
		vec3 start, dir, weight;
	};

	const int objFaces = 12;
	uniform vec3 wEye, v[20];
	uniform int planes[objFaces * 3];
	uniform int planePoints[objFaces * 5];
	uniform vec3 kd;
	uniform vec3 ks;
	uniform vec3 F0;
	uniform float deg;

	float length(vec3 p, vec3 c) {
		return sqrt(pow(p.x - c.x, 2) + pow(p.y - c.y, 2) + pow(p.z - c.z, 2));
	}

	bool isEqual(float f1, float f2) {
		if ((abs(f1 - f2) <= 0.00001f)) {
			return true;
		}
		return false;
	}

	void getObjPlane(int i, float scale, out vec3 p, out vec3 normal){
		vec3 p1 = v[planes[ 3 * i] - 1 ], p2 = v[planes[ 3 * i + 1] - 1], p3 = v[planes[ 3 * i + 2] - 1];
		normal = cross(p2 - p1, p3 - p1);
		//if(dot(p1, normal) < 0) normal = -normal;
		p = p1 * scale + vec3(0,0,0.03f);
	}

	float checkDistance(vec3 X, vec3 Y, vec3 pintersect){
		vec3 t = pintersect - X - normalize(Y - X) * (dot(pintersect - X, normalize(Y - X)));
		return dot(pintersect - X, normalize(t));
	}

	Hit intersectConvexPolyhedron(Ray ray, Hit hit, float scale){
		for(int i = 0; i < objFaces; i++){
			vec3 p1, normal;
			getObjPlane(i,scale,p1,normal);
			float ti = abs(dot(normal, ray.dir)) > epsilon ? dot(p1 - ray.start, normal) / dot(normal, ray.dir) : -1;
			if(ti <= epsilon || (ti > hit.t && hit.t > 0)) continue;
			vec3 pintersect = ray.start + ray.dir * ti;
			bool outside = false;
			for(int j = 0; j < objFaces; j++){
				if(i == j) continue;
				vec3 p11, n;
				getObjPlane(j,scale,p11,n);
				if(dot(n, pintersect -p11) > 0) {
					outside = true;
					break;
				}
			}
			
			if(!outside){
				float d = 0.15;
				float temp;
				for(int ii = 0; ii < 5; ii++){
					temp = checkDistance(v[planePoints[ii + i * 5] - 1] * scale, 
								 v[planePoints[1 + ii + i * 5] - 1] * scale, pintersect);
					if(temp <= d) d = temp;
				}
				temp = checkDistance(v[planePoints[i * 5] - 1] * scale, 
								v[planePoints[i * 5 + 4] - 1] * scale, pintersect);
				if(temp <= d) d = temp;

				hit.t = ti;
				hit.position = pintersect;
				hit.normal = normalize(normal);
				if(d <= 0.1) hit.mat = 0;
				if(d > 0.1) hit.mat = 2;
			}
		}
		return hit;
	}

	Hit intersectImplicit(Ray ray, Hit hit){
		vec3 center = vec3(0.0f, 0.0f, 0.0f);
		float radius = 0.3;
		float a = 7.0;
		float b = 7.0;
		float c = 3.0;

		float A = a * ray.dir.x * ray.dir.x + b * ray.dir.y * ray.dir.y;
		float B = 2.0 * a * ray.start.x * ray.dir.x + 2.0 * b * ray.start.y * ray.dir.y - c * ray.dir.z;
		float C = a * ray.start.x * ray.start.x + b * ray.start.y * ray.start.y - c * ray.start.z;	

		float discr = B * B - 4.0 * A * C;
		if (discr < 0) return hit;
		float sqrt_discr = sqrt(discr);
		float t1 = (-B + sqrt_discr) / 2.0f / A;
		float t2 = (-B - sqrt_discr) / 2.0f / A;

		vec3 p1 = ray.start + ray.dir * t1;
		vec3 p2 = ray.start + ray.dir * t2;

		if (length(p1, center) > radius && length(p2, center) > radius) {
			return hit;
		}

		if (length(p1, center) > radius && length(p2, center) < radius) {
			hit.position = p2;
			hit.t = t2;
		}
		if (length(p1, center) < radius && length(p2, center) > radius) {
			hit.position = p1;
			hit.t = t1;
		}
		if (length(p1, center) < radius && length(p2, center) < radius) {
			if(length(p1,ray.start) < length(p2,ray.start)) { hit.position = p1; hit.t = t1;}
			else { hit.position = p2; hit.t = t2;}
		}

		vec3 temp = ray.start + ray.dir * hit.t;

		vec3 fx = vec3(1.0,0.0, 2.0 * a * temp.x / c);
		vec3 fy = vec3(0.0,1.0, 2.0 * b * temp.y / c);
		hit.normal = normalize(cross(fx,fy));

		if (isEqual(dot(hit.normal, ray.dir), 0.0)) {
			hit.normal = -hit.normal;
		}

		hit.mat = 1;
		return hit;
	}

	Hit firstIntersect(Ray ray){
		Hit bestHit;
		bestHit.t = -1;
		bestHit = intersectImplicit(ray, bestHit);
		bestHit = intersectConvexPolyhedron(ray, bestHit, sqrt(3.0f));
		if(dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	vec3 Fresnel(vec3 F0, float cosTheta) { 
		return F0 + (vec3(1.0, 1.0, 1.0) - F0) * pow(cosTheta, 5);
	}

	vec4 qmul(vec4 q1, vec4 q2){
		vec3 d1 = vec3(q1.x, q1.y, q1.z);
		vec3 d2 = vec3(q2.x, q2.y, q2.z);
		return vec4(d2 * q1.w + d1 * q2.w + cross(d1, d2), q1.w * q2.w - dot(d1,d2));
	}
	
	vec4 quaternion(float ang, vec3 axis) {
		vec3 d = normalize(axis) * sin(ang / 2);
		return vec4(d.x, d.y, d.z, cos(ang / 2));
	}

	vec3 Rotate(vec3 u, vec4 q){
		vec4 qinv  = vec4(-q.x, -q.y, -q.z, q.w);
		vec4 qr = qmul(qmul(q, vec4(u.x, u.y, u.z, 0)), qinv);
		return vec3(qr.x, qr.y, qr.z);
	}

	vec3 trace(Ray ray){
		vec3 outRadiance = vec3(0.0,0.0,0.0);
		for(int d = 0; d < maxdepth; d++){
			Hit hit = firstIntersect(ray);
			if(hit.t < 0) break;
			if(hit.mat == 0) {
				vec3 lightdir = normalize(lightPosition - hit.position);
				float cosTheta = dot(hit.normal, lightdir);
				if(cosTheta > 0){
					vec3 LeIn = Le / dot(lightPosition - hit.position, lightPosition - hit.position);
					outRadiance += ray.weight * LeIn * kd * cosTheta;
					vec3 halfway = normalize(-ray.dir + lightdir);
					float cosDelta = dot(hit.normal, halfway);
					if(cosDelta > 0) outRadiance += ray.weight * LeIn * ks * pow(cosDelta, shininess);
				}
				ray.weight *= ka;
				break;
			}
			if(hit.mat == 1){
				ray.weight *= Fresnel(F0, 1.0f - dot(-ray.dir, hit.normal));
				ray.start = hit.position + hit.normal * epsilon;
				ray.dir = reflect(ray.dir, hit.normal);
			}
			if(hit.mat == 2){
				vec4 q = quaternion(deg, hit.normal);
				ray.start = hit.position + hit.normal * epsilon;
				ray.dir = reflect(ray.dir, hit.normal);
				ray.start = Rotate(ray.start, q);
				ray.dir = Rotate(ray.dir, q);
			}
		}
		outRadiance += ray.weight * La;
		return outRadiance;
	}

	in vec3 p;
	out vec4 fragmentColor;

	void main(){
		Ray ray;
		ray.start = wEye;
		ray.dir = normalize(p - wEye);
		ray.weight = vec3(1,1,1);
		fragmentColor = vec4(trace(ray),1);
	}
)";

struct Camera {
	vec3 eye, lookat, right, pvup, rvup;
	float fov = 45 * (float)M_PI / 180;

	Camera() : eye(0, 1, 1), pvup(0, 0, 1), lookat(0, 0, 0) { set(); }
public:
	void set() {
		vec3 w = eye - lookat;
		float f = length(w);
		right = normalize(cross(pvup, w)) * f * tanf(fov / 2);
		rvup = normalize(cross(w, right)) * f * tanf(fov / 2);
	}
	void Animate(float t) {
		float r = sqrtf(eye.x * eye.x + eye.y * eye.y);
		eye = vec3(r * cos(t) + lookat.x, r * sin(t) + lookat.y, eye.z);
		set();
	}
};

GPUProgram shader;
Camera camera;

float F(float n, float k) { return ((n - 1) * (n - 1) + k * k) / ((n + 1) * (n + 1) + k * k); }

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	unsigned int vao, vbo;
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	float vertexCoords[] = { -1,-1,1,-1,1,1,-1,1 };
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

	shader.create(vertexSource, fragmentSource, "fragmentColor");

	const float g = 0.618f, G = 1.618f;
	std::vector <vec3> v = {
			vec3(0, g, G), vec3(0, -g, G), vec3(0, -g, -G), vec3(0, g, -G), vec3(G, 0, g),
			vec3(-G, 0, g), vec3(-G, 0, -g), vec3(G, 0, -g), vec3(g, G, 0), vec3(-g, G, 0),
			vec3(-g, -G, 0), vec3(g, -G, 0), vec3(1, 1, 1), vec3(-1, 1, 1), vec3(-1, -1, 1), 
			vec3(1, -1, 1),vec3(1, -1, -1), vec3(1, 1, -1), vec3(-1, 1, -1), vec3(-1, -1, -1)
	};
	for (int i = 0; i < v.size(); i++) {
		shader.setUniform(v[i], "v[" + std::to_string(i) + "]");
	}

	std::vector<int> planes = {
				1, 2, 16,
				1, 13, 9, 
				1, 14, 6, 
				2, 15, 11, 
				3, 4, 18, 
				3, 17, 12, 
				3, 20, 7,
				19, 10, 9, 
				16, 12, 17,
				5, 8, 18,
				14, 10, 19, 
				6, 7, 20
	};
	for (int i = 0; i < planes.size(); i++) {
		shader.setUniform(planes[i], "planes[" + std::to_string(i) + "]");
	}

	std::vector<int> planePoints = {
			1, 2, 16, 5, 13, 
			1, 13, 9, 10, 14, 
			1, 14, 6, 15, 2, 
			2, 15, 11, 12, 16,
			3, 4, 18, 8, 17, 
			3, 17, 12, 11, 20,
		    3, 20, 7, 19, 4,
			19, 10, 9, 18, 4,
			16, 12, 17, 8, 5,
			5, 8, 18, 9, 13,
			14, 10, 19, 7, 6,
			6, 7, 20, 11, 15
	};
	for (int i = 0; i < planePoints.size(); i++) {
		shader.setUniform(planePoints[i], "planePoints[" + std::to_string(i) + "]");
	}

	shader.setUniform(vec3(1.9f, 0.6f, 0.4f), "kd");
	shader.setUniform(vec3(5, 5, 5), "ks");
	shader.setUniform(vec3(F(0.17, 3.1), F(0.35, 2.7), F(1.5, 1.9)), "F0");
	float deg = 72 * (float)M_PI / 180;
	shader.setUniform(deg, "deg");
}


void onDisplay() {
	glClearColor(0, 0, 0, 0);							
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 

	shader.setUniform(camera.eye, "wEye");
	shader.setUniform(camera.lookat, "wLookAt");
	shader.setUniform(camera.right, "wRight");
	shader.setUniform(camera.rvup, "wUp");

	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) { }

void onKeyboardUp(unsigned char key, int pX, int pY) {}

void onMouseMotion(int pX, int pY) {}

void onMouse(int button, int state, int pX, int pY) {}

void onIdle() {
	camera.Animate(glutGet(GLUT_ELAPSED_TIME) / 1000.0f);
	glutPostRedisplay();
}
