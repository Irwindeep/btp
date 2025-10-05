#include "desertsim/Code/Include/desert.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <ostream>
#include <sstream>
#include <vector>
#include <string>
#include <filesystem>
namespace fs = std::filesystem;

struct DuneConfig {
	float rMin;
	float rMax;
	float windX;
	float windY;
	int vegetation;
	int abrasion;
	int preSimSteps;
};

int main() {
	std::ifstream file("data_config.csv");
	if (!file.is_open()) {
		std::cerr << "Could not open file\n";
		return 1;
	}

	std::string saveDir = "dunesim_dataset/";

	if (!fs::exists(saveDir)){
		if (!fs::create_directories(saveDir)) {
			std::cerr << "Failed to create directory: " << saveDir << "\n";
			return 1;
		}
	}

	std::string line;
	// Read the header line and ignore
	std::getline(file, line);

	std::vector<DuneConfig> configs;

	while (std::getline(file, line)) {
		std::stringstream ss(line);
		std::string cell;
		DuneConfig cfg;

		std::getline(ss, cell, ',');
		cfg.rMin = std::stof(cell);
		std::getline(ss, cell, ',');
		cfg.rMax = std::stof(cell);
		std::getline(ss, cell, ',');
		cfg.windX = std::stof(cell);
		std::getline(ss, cell, ',');
		cfg.windY = std::stof(cell);
		std::getline(ss, cell, ',');
		cfg.vegetation = std::stof(cell);
		std::getline(ss, cell, ',');
		cfg.abrasion = std::stof(cell);
		std::getline(ss, cell, ',');
		cfg.preSimSteps = std::stof(cell);

		configs.push_back(cfg);
	}

	std::cout << "Total configs read: " << configs.size() << "\n";
	
	int idx = 0;
	for(const auto& config: configs){
		DuneSediment dune = DuneSediment(
			Box2D(Vector2(0), Vector2(256)),
			config.rMin,
			config.rMax,
			Vector2(config.windX, config.windY)
		);
		if(config.vegetation) dune.SetVegetationMode(true);
		if(config.abrasion) dune.SetAbrasionMode(true);
		for(int i = 0; i < config.preSimSteps; i++)
			dune.SimulationStepMultiThreadAtomic();

		std::ostringstream fname_base;
		fname_base << std::setw(4) << std::setfill('0') << idx << '/';

		if (!fs::exists(saveDir + fname_base.str())){
			if (!fs::create_directories(saveDir + fname_base.str())) {
				std::cerr << "Failed to create directory: " << saveDir + fname_base.str() << "\n";
				return 1;
			}
		}

		{
			std::ofstream out(saveDir + fname_base.str() + "bedrock0.raw", std::ios::binary);
			for (int j = 0; j < 256; ++j)
				for (int i = 0; i < 256; ++i) {
					float v = dune.Bedrock(i, j);
					out.write(reinterpret_cast<const char*>(&v), sizeof(float));
				}
		}

		{
			std::ofstream out(saveDir + fname_base.str() + "sediments0.raw", std::ios::binary);
			for (int j = 0; j < 256; ++j)
				for (int i = 0; i < 256; ++i) {
					float v = dune.Sediment(i, j);
					out.write(reinterpret_cast<const char*>(&v), sizeof(float));
				}
		}

		dune.SimulationStepMultiThreadAtomic();

		{
			std::ofstream out(saveDir + fname_base.str() + "bedrock1.raw", std::ios::binary);
			for (int j = 0; j < 256; ++j)
				for (int i = 0; i < 256; ++i) {
					float v = dune.Bedrock(i, j);
					out.write(reinterpret_cast<const char*>(&v), sizeof(float));
				}
		}

		{
			std::ofstream out(saveDir + fname_base.str() + "sediments1.raw", std::ios::binary);
			for (int j = 0; j < 256; ++j)
				for (int i = 0; i < 256; ++i) {
					float v = dune.Sediment(i, j);
					out.write(reinterpret_cast<const char*>(&v), sizeof(float));
				}
		}

		if(idx%1000 == 0)
			std::cout << "Generated " << idx+1 << " Datapoints" << std::endl;

		idx++;
	}
}
