# Autonoma

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![MCP](https://img.shields.io/badge/protocol-MCP-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

**State-of-the-art Text-to-CAD AI agent that converts natural language into complex 3D CAD models of real hardware.**

Autonoma is the most capable open-source AI agent for 3D mechanical design. Describe what you want to build in plain English, and the AI agent designs it for you in real-time. The vision: **democratize robotics and hardware the way vibecoding has democratized programming.**

The closest competitor is [Zoo.dev's Text-to-CAD](https://zoo.dev/text-to-cad) -- closed-source and commercialized. Autonoma is open-source and, I believe, more capable.

> Autonoma received an investment offer of **$130K SAFE at $5M valuation** from [Humanoid Global Holdings](https://www.humanoidglobal.ai/) ($130K for 2.6% equity).

## Key Features

- **300-500 CAD modeling operations** across 9 specialized AI tools
- **SoTA visual feedback system** -- the AI receives rendered feedback from **10 different view angles** plus detailed textual topology analysis to self-correct its designs
- **Real-time 3D viewer** -- watch the AI build your model live and interrupt if it makes a mistake
- **Pre-built parts library** -- 30+ common components (Arduino, Raspberry Pi, motors, sensors, batteries, wheels, grippers) ready to import
- **Intelligent context management** -- on-demand documentation loading system so the AI always has the right reference for the operation at hand
- **Full session state management** -- persistent object storage, hierarchy tracking, operation history
- **STEP import/export** -- industry-standard format for use in any CAD software

## Demo

### Video [Click the image]

[![Demo Video - AI agents creating a 3D gear from scratch](demo/gear.png)](https://youtu.be/TlQVlgVZJVA)

### Pictures

![Robotic Arm Claw](demo/robotic-arm-claw.png)
![Complex 3D Model](demo/complex-3d-model.png)
![Chassis for Robotic Car](demo/chassis.png)
![Mount for Ultrasonic Sensor](demo/ultrasonic-sensor-mount.png)
![Long Nut](demo/long-nut.png)
![Servo Motor](demo/servo-motor.png)
![Alien-Technology](demo/alien-tech.png)
![Deformed Robotic Car #1](demo/car-1.png)
![Deformed Robotic Car #2](demo/car-2.png)

## Architecture

```
                          ┌──────────────────────┐
                          │   User (Text Prompt) │
                          └──────────┬───────────┘
                                     │
                                     ▼
                          ┌──────────────────────┐
                          │   LLM (AI Agent)     │
                          │  (Claude, GPT, etc.) │
                          └──────────┬───────────┘
                                     │  structured tool calls
                                     ▼
┌────────────────────────────────────────────────────────────────────┐
│                     Autonoma Server (MCP)                          │
│                                                                    │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │
│  │ fluent_api  │ │ direct_api  │ │geometry_api │ │selector_api │   │
│  │ (Workplane, │ │ (Edge,Wire, │ │(Vector,Plane│ │(20+ selector│   │
│  │  Sketch,    │ │  Face,Solid)│ │ Location,   │ │  classes)   │   │
│  │  Assembly)  │ │             │ │ Matrix)     │ │             │   │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘   │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │
│  │free_function│ │state_manager│ │import_export│ │  load_docs  │   │
│  │  (stateless │ │ (objects,   │ │(STEP files, │ │(on-demand   │   │
│  │   shapes)   │ │  history)   │ │ parts lib)  │ │ API docs)   │   │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘   │
│  ┌─────────────┐                                                   │
│  │  feedback   │──── 10-angle visual + textual topology analysis   │
│  └─────────────┘                                                   │
│                                                                    │
└─────────────────────────────────────┬──────────────────────────────┘
                                      │
                          ┌───────────▼──────────┐
                          │    OpenCASCADE       │
                          │  (CAD Kernel / BREP) │
                          └───────────┬──────────┘
                                      │
                          ┌───────────▼──────────┐
                          │  Real-time 3D Viewer │
                          │  (OCP CAD Viewer)    │
                          └──────────────────────┘
```

Single-file server (~3,500 lines) that gives AI agents complete programmatic control over the OpenCASCADE CAD kernel. The AI doesn't generate code -- it directly executes CAD operations through structured tool calls, enabling precise control and real-time feedback loops that code-generation approaches cannot match.

## Known Limitations

- Spatial positioning of parts in assemblies can be imprecise (solvable)
- AI agent occasionally misinterprets visual/textual feedback (solvable)

## About

I'm **Katif**, an AI and robotics engineer with 5 years of experience. I built Autonoma as a solo founder (CEO & CTO) to prove that AI agents can design complete, real-world robots from text alone.

**Target market:** Mechanical engineering design -- $10B market (2024), 45M potential users. The goal was to use AI to fully disrupt this space by replacing manual CAD work with AI agents, eliminating the learning curve and cost barrier of robotics.

I'm open-sourcing the complete codebase now that the startup has shut down. I hope others will build on this work.

## License

MIT — see [LICENSE](LICENSE) for details.
