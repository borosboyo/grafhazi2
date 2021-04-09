//=============================================================================================
// Computer Graphics Sample Program: Ray-tracing-let
//=============================================================================================
#include "framework.h"


const float epsilon = 0.00001f;
const vec3 n = vec3(0.17, 0.35, 1.5);
const vec3 k = vec3(3.1, 2.7, 1.9);

float F(float n, float k) { return ((n - 1) * (n - 1) + k * k) / ((n + 1) * (n + 1) + k * k); }

struct Material {
	vec3 ka, kd, ks;
	float  shininess;
	vec3 F0;
	int rough, reflective;
};


vec3 operator/(vec3 num, vec3 denom) {
	return vec3(num.x / denom.x, num.y / denom.y, num.z / denom.z);
}


struct RoughMaterial : Material {
	RoughMaterial(vec3 _kd, vec3 _ks, float _shininess) {
		ka = _kd * M_PI;
		kd = _kd;
		ks = _ks;
		shininess = _shininess;
		rough = true;
		reflective = false;
	}
};

struct ReflectiveMaterial : Material {
	ReflectiveMaterial(vec3 _F0) {
		F0 = _F0;
		rough = false;
		reflective = true;
	}
};

struct Hit {
	float t;
	vec3 position, normal;
	Material* material;
	Hit() { t = -1; }
};

struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};

class Intersectable {
protected:
	Material* material;
public:
	virtual Hit intersect(const Ray& ray) = 0;
};


float length(vec3 p, vec3 c) {
	return sqrtf(powf(p.x - c.x, 2) + powf(p.y - c.y, 2) + powf(p.z - c.z, 2));
}

bool isEqual(float f1, float f2) {
	if ((fabsf(f1 - f2) <= epsilon)) {
		return true;
	}
	return false;
}

struct Implicit : public Intersectable {
	vec3 center = vec3(0.0f,0.0f,0.0f);
	float radius = 0.3f;
	float a = 8.0f;
	float b = 8.0f;
	float c = 3.0f;
	//Kivenni a dolgokat
	Implicit(const vec3& _center, float _radius, Material* _material) {
		center = _center;
		radius = _radius;
		material = _material;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;

		float A = a * ray.dir.x * ray.dir.x + b * ray.dir.y * ray.dir.y;
		float B = 2.0f * a * ray.start.x * ray.dir.x + 2.0f * b * ray.start.y * ray.dir.y - c * ray.dir.z;
		float C = a * ray.start.x * ray.start.x + b * ray.start.y * ray.start.y - c * ray.start.z;

		float discr = B * B - 4.0 * A * C;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-B + sqrt_discr) / 2.0 / A;
		float t2 = (-B - sqrt_discr) / 2.0 / A;
		if (t1 <= 0) return hit;

		vec3 p1;
		p1 = ray.start + ray.dir * t1;
		//p1.y = ray.start.y + ray.dir.y * t1;
		//p1.z = ray.start.z + ray.dir.z * t1;



		vec3 p2;
		p2 = ray.start + ray.dir * t2;
		//p2.y = ray.start.y + ray.dir.y * t2;
		//p2.z = ray.start.z + ray.dir.z * t2;




		//pont annyi lehet nemkell

		if (length(p1, center) > radius && length(p2, center) > radius) {
			return hit;
		}
		if (length(p1, center) > radius && length(p2, center) < radius) {
			if (t2 > 0) {
				hit.t = t2;
			}
		}
		if (length(p1, center) < radius && length(p2, center) > radius) {
			hit.t = t1;
		}
		if (length(p1, center) < radius && length(p2, center) < radius) {
			hit.t = (t2 > 0) ? t2 : t1;
		}


		float X = ray.start.x + ray.dir.x * hit.t;
		float Y = ray.start.y + ray.dir.y * hit.t;
		float Z = ray.start.z + ray.dir.z * hit.t;


		//vec3 fx = (1.0, 0.0, (2.0 * a * X) / c);
		//vec3 fy = (0.0, 1.0, (2.0 * b * Y) / c);

		//vec3 n = cross(fx, fy);



		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = vec3(-((2.0 * a * X) / c), -((2.0 * b * Y) / c), 1.0);
		

		if (isEqual(dot(hit.normal, ray.start), 0.0)) {
			hit.normal = -hit.normal;
		}
		
		hit.material = material;
				
		return hit;
	}
};


struct Test: public Intersectable {
	vec3 center;
	float radius = 0.3f;
	float a = 6.0f;
	float b = 6.0f;
	float c = 1.0f;
	//Kivenni a dolgokat
	Test(const vec3& _center, float _radius, Material* _material) {
		center = _center;
		radius = _radius;
		material = _material;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 dist = ray.start - center;
		float a = dot(ray.dir, ray.dir);
		float b = dot(dist, ray.dir) * 2.0f;
		float c = dot(dist, dist) - radius * radius;
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = (hit.position - center) * (1.0f / radius);
		hit.material = material;
		return hit;
	}
};


struct Dodeka : public Intersectable {
	std::vector<vec3> v;
	int objFaces = 12;
	std::vector<int> planes;

	Dodeka(Material* _material) {
		material = _material;
		const float g = 0.618f, G = 1.618f;
		v = {
			vec3(0,g,G), vec3(0,-g,G), vec3(0,-g,-G), vec3(0,g,-G), vec3(G,0,g),vec3(-G,0,g), vec3(-G,0,-g),
			vec3(G,0,-g), vec3(g,G,0), vec3(-g,G,0), vec3(-g,-G,0), vec3(g,-G,0), vec3(1, 1, 1), vec3(-1, 1, 1),
			vec3(-1, -1, 1), vec3(1, -1, 1), vec3(1, -1, -1), vec3(1, 1, -1), vec3(-1, 1, -1), vec3(-1, -1, -1)
		};

		planes = {
			1,2,16,   1,13,9,   1,14,6,      2,15,11,  3,4,18,   3,17,12,
			3,20,7,   19,10,9,   16,12,17,   5,8,18,  14,10,19,  6,7,20
		};

	}

	void getObjPlane(int i, vec3 p, vec3 normal) {
		vec3 p1 = v[planes[3 * i] - 1];
		vec3 p2 = v[planes[3 * i + 1] - 1];
		vec3 p3 = v[planes[3 * i + 2] - 1];
		normal = cross(p2 - p1, p3 - p1);
		if (dot(p1, normal) < 0) normal = -normal;
		p = p1 + vec3(0, 0, 0.03f);
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		for (int ii = 0; ii < objFaces; ii++) {
			vec3 p1, normal;
			getObjPlane(ii, p1, normal);
			float ti = abs(dot(normal, ray.dir)) > epsilon ? dot(p1 - ray.start, normal) / dot(normal, ray.dir) : -1;
			if (ti <= epsilon || (ti > hit.t && hit.t > 0)) continue;
			vec3 pintersect = ray.start + ray.dir * ti;
			bool outside = false;
			for (int jj = 0; jj < objFaces; jj++) {
				if (ii == jj) continue;
				vec3 p11, n;
				getObjPlane(jj, p11, n);
				if (dot(n, pintersect - p11) > 0) {
					outside = true;
					break;
				}
			}
			if (!outside) {
				hit.t = ti;
				hit.position = pintersect;
				hit.normal = normalize(normal);
				hit.material = material;
			}
		}
		return hit;
	}


};


class Camera {
	vec3 eye, lookat, right, up;
	float fov;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float _fov) {
		eye = _eye;
		lookat = _lookat;
		fov = _fov;
		vec3 w = eye - lookat;
		float focus = length(w);
		right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
		up = normalize(cross(w, right)) * focus * tanf(fov / 2);
	}
	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}

	void Animate(float dt) {
		eye = vec3((eye.x - lookat.x) * cos(dt) + (eye.z - lookat.z) * sin(dt) + lookat.x,
			eye.y,
			-(eye.x - lookat.x) * sin(dt) + (eye.z - lookat.z) * cos(dt) + lookat.z);
		set(eye, lookat, up, fov);
	}

};

struct Light {
	vec3 direction;
	vec3 Le;
	Light(vec3 _direction, vec3 _Le) {
		direction = normalize(_direction);
		Le = _Le;
	}
};

float rnd() { return (float)rand() / RAND_MAX; }


class Scene {
	std::vector<Intersectable*> objects;
	std::vector<Light*> lights;
	Camera camera;
	vec3 La;
public:
	void build() {
		vec3 eye = vec3(0, 0, 2), vup = vec3(0, 2, 0), lookat = vec3(0, 0, 0);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		La = vec3(1.0f, 1.0f, 1.0f);
		vec3 lightDirection(1, 1, 1), Le(2, 2, 2);
		lights.push_back(new Light(lightDirection, Le));

		//Rough
		vec3 kd(0.3f, 0.2f, 0.1f), ks(2, 2, 2);
		Material* mat2 = new RoughMaterial(kd, ks, 50);


		vec3 F0;
		F0.x = ((n.x - 1.0) * (n.x - 1.0) + k.x * k.x) / ((n.x + 1.0) * (n.x + 1.0) + k.x * k.x);
		F0.y = ((n.y - 1.0) * (n.y - 1.0) + k.y * k.y) / ((n.y + 1.0) * (n.y + 1.0) + k.y * k.y);
		F0.z = ((n.z - 1.0) * (n.z - 1.0) + k.z * k.z) / ((n.z + 1.0) * (n.z + 1.0) + k.z * k.z);
		//printf("%3.20f %3.20f %3.20f", F0.x, F0.y, F0.z);
		Material* mat = new ReflectiveMaterial(F0);




		// ARANY MATERIAL
		// ALAKZAT KINÉZETE X 
		// DODEKAÉDER
		// DODEKAÉDER SAKRAI
		// PORTÁLOK
		objects.push_back(new Implicit(vec3(0.0f, 0.0f, 0.0f), 0.3f, mat));
		for (int i = 0; i < 14; i++) {
			//objects.push_back(new Sphere(vec3(rnd() - 0.5f, rnd() - 0.5f, rnd() - 0.5f), rnd() * 0.1f, mat));
			//objects.push_back(new Sphere(vec3(0.5f,  0.5f, 0.5f), 0.3f, mat));
			//objects.push_back(new Sphere(vec3(rnd() - 0.5f, rnd() - 0.5f, rnd() - 0.5f), rnd() * 0.1f, mat));
			//objects.push_back(new Sphere(vec3(rnd() - 0.5f, rnd() - 0.5f, rnd() - 0.5f), rnd() * 0.1f, mat2));
	
			//objects.push_back(new Test(vec3(rnd() - 0.5f, rnd() - 0.5f, rnd() - 0.5f), rnd() * 0.1f, mat2));
		}

		//objects.push_back(new Dodeka(mat2));
	}

	void render(std::vector<vec4>& image) {
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable* object : objects) {
			Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	bool shadowIntersect(Ray ray) {	// for directional lights
		for (Intersectable* object : objects) if (object->intersect(ray).t > 0) return true;
		return false;
	}

	vec3 trace(Ray ray, int depth = 0) {
		if (depth > 5)
			return La;

		Hit hit = firstIntersect(ray);
		if (hit.t < 0)
			return La;

		vec3 outRadiance = (0, 0, 0);
		if (hit.material->rough == true) {
			outRadiance = hit.material->ka * La;
			for (Light* light : lights) {
				Ray shadowRay(hit.position + hit.normal * epsilon, light->direction);
				float cosTheta = dot(hit.normal, light->direction);
				if (cosTheta > 0 && !shadowIntersect(shadowRay)) {	// shadow computation
					outRadiance = outRadiance + light->Le * hit.material->kd * cosTheta;
					vec3 halfway = normalize(-ray.dir + light->direction);
					float cosDelta = dot(hit.normal, halfway);
					if (cosDelta > 0) outRadiance = outRadiance + light->Le * hit.material->ks * powf(cosDelta, hit.material->shininess);
				}
			}
		}

		if (hit.material->reflective == true) {
			vec3 reflectedDir = ray.dir - hit.normal * dot(hit.normal, ray.dir) * 2.0f;
			float cosa = -dot(ray.dir, hit.normal);
			vec3 F;
			//F = hit.material->F0 + (one - hit.material->F0) * pow(1 - cosa, 5);
			F.x = hit.material->F0.x + (1.0 - hit.material->F0.x) * pow(1.0 - cosa, 5);
			F.y = hit.material->F0.y + (1.0 - hit.material->F0.y) * pow(1.0 - cosa, 5);
			F.z = hit.material->F0.z + (1.0 - hit.material->F0.z) * pow(1.0 - cosa, 5);
			//printf("%3.2f %3.2f %3.2f \n ", F.x, F.y, F.z);
			outRadiance = outRadiance + trace(Ray(hit.position + hit.normal * epsilon, reflectedDir), depth + 1) * F;
		}

		return outRadiance;
	}

	void Animate(float dt) {
		camera.Animate(dt);
	}
};

GPUProgram gpuProgram; // vertex and fragment shaders
Scene scene;

// vertex shader in GLSL
const char* vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char* fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";

class FullScreenTexturedQuad {
	unsigned int vao;	// vertex array object id and texture id
	Texture texture;
public:

	FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image)
		: texture(windowWidth, windowHeight, image)
	{
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

		// vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed
	}


	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		gpuProgram.setUniform(texture, "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};

FullScreenTexturedQuad* fullScreenTexturedQuad;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();

	std::vector<vec4> image(windowWidth * windowHeight);
	long timeStart = glutGet(GLUT_ELAPSED_TIME);
	scene.render(image);
	long timeEnd = glutGet(GLUT_ELAPSED_TIME);
	printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));

	// copy image to GPU as a texture
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen

	std::vector<vec4> image(windowWidth * windowHeight);
	scene.render(image);
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	scene.Animate(0.07f);
	glutPostRedisplay();
}