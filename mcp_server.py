"""
CadQuery MCP Server - A FastMCP server providing complete control over the CadQuery Python library.

This server provides 6 tools for 3D CAD modeling:
- cq_fluent_api: High-level chainable operations (Workplane, Sketch, Assembly)
- cq_direct_api: Low-level geometry construction (Edge, Wire, Face, Solid, etc.)
- cq_geometry_api: Mathematical primitives (Vector, Plane, Location, Matrix, BoundBox)
- cq_selector_api: Object selection and filtering (20+ selector classes)
- cq_free_function_api: Free function API (stateless shape construction)
- cq_state_manager: Session state management (list, delete, undo/redo, etc.)

State Management:
Uses module-level global state (_session_state) to persist objects across MCP requests,
since FastMCP's Context.state is request-scoped.

Transport: stdio (local subprocess)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Literal
import copy
import json
import os
import tempfile
import traceback
import sys

# Try to import OCP viewer - may not be available in all environments
try:
    from ocp_vscode import show, set_port
    set_port(3939)
    OCP_VIEWER_AVAILABLE = True
except ImportError:
    OCP_VIEWER_AVAILABLE = False
    show = None

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp import Image as MCPImage

# Import CadQuery
import cadquery as cq
from cadquery import (
    Workplane, Sketch, Assembly,
    Vector, Plane, Location, Matrix, BoundBox,
    Vertex, Edge, Wire, Face, Shell, Solid, Compound,
    Color, exporters
)
from cadquery.selectors import (
    Selector, NearestToPointSelector, BoxSelector, BaseDirSelector,
    ParallelDirSelector, DirectionSelector, PerpendicularDirSelector,
    TypeSelector, RadiusNthSelector, CenterNthSelector,
    DirectionMinMaxSelector, DirectionNthSelector, LengthNthSelector,
    AreaNthSelector, BinarySelector, AndSelector, SumSelector,
    SubtractSelector, InverseSelector, StringSyntaxSelector
)

# Try to import free functions (may not be available in all versions)
try:
    from cadquery.func import (
        edgeOn, wireOn, wire, face,  shell, solid, compound, vertex, segment, polyline, polygon, rect, spline, circle, ellipse, plane, box, cylinder, sphere, torus, cone, text, fuse, cut, intersect, imprint, split, fill, clean, cap, fillet, chamfer, extrude, revolve, offset, sweep, loft, check, closest, setThreads, project, faceOn, isSubshape
    )
    FREE_FUNCTIONS_AVAILABLE = True
except ImportError:
    FREE_FUNCTIONS_AVAILABLE = False

# PNG conversion library availability check (for cq_feedback tool)
_PNG_CONVERSION_AVAILABLE = False
_PNG_LIBRARY = None

# Try playwright (primary - Windows compatible, faster than selenium)
try:
    from playwright.async_api import async_playwright
    from PIL import Image
    import io
    import base64
    import nest_asyncio
    _PNG_CONVERSION_AVAILABLE = True
    _PNG_LIBRARY = "playwright"
except ImportError:
    # Try svglib fallback (pure Python)
    try:
        from svglib.svglib import svg2rlg
        from reportlab.graphics import renderPM
        from PIL import Image
        import io
        import base64
        _PNG_CONVERSION_AVAILABLE = True
        _PNG_LIBRARY = "svglib+pillow"
    except ImportError:
        # base64 always available, just import for later use
        import base64
        pass

# Initialize FastMCP server
mcp = FastMCP(
    name="cadquery_mcp", 
    instructions="IMPORTANT: Before using any API tool (cq_fluent_api, cq_direct_api, cq_geometry_api, cq_selector_api, cq_free_function_api), you MUST first call the cq_load_docs tool to load the relevant documentation for the API you want to use. IMPORTANT: Before using cq_load_docs tool to get the documentation of an API tool, you MUST first use it to get the overview documentation, because it can help you decide which API tool to use.")

# Constants
MAX_UNDO_STACK_SIZE = 50

# Debug mode control for traceback exposure
DEBUG_MODE = True #os.environ.get("CQ_MCP_DEBUG", "false").lower() == "true"

# Standard named views for multi-angle rendering (for cq_feedback tool)
STANDARD_VIEWS = {
    # Orthographic views
    "front": (0, -1, 0),
    "back": (0, 1, 0),
    "left": (-1, 0, 0),
    "right": (1, 0, 0),
    "top": (0, 0, 1),
    "bottom": (0, 0, -1),

    # Four isometric corner views
    "isometric_front_right": (3, -2, 1),
    "isometric_front_left": (-3, -2, 1),
    "isometric_back_right": (3, 2, 1),
    "isometric_back_left": (-3, 2, 1)
}

# =============================================================================
# ERROR HANDLING
# =============================================================================

def error_response(error_msg: str, tb: str = None) -> str:
    """Create standardized error response with optional traceback.

    Tracebacks are only included when CQ_MCP_DEBUG=true environment variable is set.
    This prevents exposing internal implementation details in production.
    """
    response = {"status": "error", "error": error_msg}
    if DEBUG_MODE and tb:
        response["traceback"] = tb
    return json.dumps(response)

# =============================================================================
# OCP VIEWER
# =============================================================================

def show_safe(obj: Any) -> None:
    """Safely show object in OCP viewer, redirecting stdout to stderr."""
    if not OCP_VIEWER_AVAILABLE:
        return  # Silently skip if viewer not available

    old = sys.stdout
    sys.stdout = sys.stderr
    try:
        # Convert Assembly to Compound to preserve locations
        if isinstance(obj, Assembly):
            show(obj.toCompound())
        elif isinstance(obj, Sketch):
            show(Workplane().placeSketch(obj))
        elif hasattr(obj, 'wrapped'):
            # It's a CadQuery shape (has wrapped attribute)
            show(obj)
        else:
            # Check if it's a raw OCP shape that needs wrapping
            obj_type_name = type(obj).__name__
            if obj_type_name.startswith("TopoDS_") or "OCP" in str(type(obj).__module__):
                # Wrap raw OCP shapes using CadQuery Shape class
                from cadquery import Shape
                wrapped_shape = Shape(obj)
                show(wrapped_shape)
            else:
                # Try to show directly
                show(obj)
    except Exception:
        # Silently ignore visualization errors
        pass
    finally:
        sys.stdout = old

# show_safe(Workplane("XY")) # Initializing the viewer with an empty workplane. This causes timeout so use clear_state instead!

# =============================================================================
# STATE MANAGEMENT
# =============================================================================

@dataclass
class CQObject:
    """Metadata wrapper for CadQuery objects."""
    name: str
    obj: Any  # Actual CadQuery object
    obj_type: str  # "Workplane", "Solid", "Assembly", etc.
    created_at: datetime = field(default_factory=datetime.now)
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)
    operation: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "obj_type": self.obj_type,
            "created_at": self.created_at.isoformat(),
            "parent": self.parent,
            "children": self.children,
            "operation": self.operation
        }


@dataclass
class StateSnapshot:
    """Snapshot for undo/redo functionality."""
    timestamp: datetime
    objects: Dict[str, CQObject]  # Full deep copy of CQObject instances
    current: Optional[str]
    description: str


# Module-level global state (persists across requests)
_session_state: Dict[str, Any] = {
    "objects": {},      # name -> CQObject
    "current": None,    # Active object name
    "history": [],      # Operation log
    "counters": {},     # Auto-naming counters
    "undo_stack": [],   # List[StateSnapshot] for undo
    "redo_stack": []    # List[StateSnapshot] for redo
}


class StateManager:
    """Static methods for state operations."""

    @staticmethod
    def store(name: str, obj: Any, obj_type: str, parent: Optional[str] = None,
              operation: str = "", parameters: Optional[dict] = None) -> str:
        """Store an object with metadata."""
        if parameters is None:
            parameters = {}

        # Create CQObject wrapper
        cq_obj = CQObject(
            name=name,
            obj=obj,
            obj_type=obj_type,
            parent=parent,
            operation=operation,
            parameters=parameters
        )

        # Store in session state
        _session_state["objects"][name] = cq_obj
        _session_state["current"] = name

        # Update parent's children list
        if parent and parent in _session_state["objects"]:
            parent_obj = _session_state["objects"][parent]
            if name not in parent_obj.children:
                parent_obj.children.append(name)

        # Add to history
        _session_state["history"].append({
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "name": name,
            "obj_type": obj_type
        })

        return name

    @staticmethod
    def get(name: str = None) -> Any:
        """Retrieve an object (or current if name is None)."""
        if name is None:
            name = _session_state["current"]
        if name is None:
            raise ValueError("No object name specified and no current object set")
        if name not in _session_state["objects"]:
            raise ValueError(f"Object '{name}' not found")
        return _session_state["objects"][name].obj

    @staticmethod
    def get_metadata(name: str) -> CQObject:
        """Get metadata for an object."""
        if name not in _session_state["objects"]:
            raise ValueError(f"Object '{name}' not found")
        return _session_state["objects"][name]

    @staticmethod
    def delete(name: str, cascade: bool = False) -> None:
        """Remove an object."""
        if name not in _session_state["objects"]:
            raise ValueError(f"Object '{name}' not found")

        obj = _session_state["objects"][name]

        # Delete children if cascade
        if cascade:
            for child in obj.children.copy():
                StateManager.delete(child, cascade=True)

        # Remove from parent's children list
        if obj.parent and obj.parent in _session_state["objects"]:
            parent = _session_state["objects"][obj.parent]
            if name in parent.children:
                parent.children.remove(name)

        # Delete the object
        del _session_state["objects"][name]

        # Clear current if it was this object
        if _session_state["current"] == name:
            _session_state["current"] = None

    @staticmethod
    def list_objects(obj_type: str = None) -> List[dict]:
        """List all objects with metadata."""
        result = []
        for name, cq_obj in _session_state["objects"].items():
            if obj_type is None or cq_obj.obj_type == obj_type:
                result.append(cq_obj.to_dict())
        return result

    @staticmethod
    def auto_name(base_name: str) -> str:
        """Generate a unique name."""
        if base_name not in _session_state["counters"]:
            _session_state["counters"][base_name] = 0
        _session_state["counters"][base_name] += 1
        return f"{base_name}_{_session_state['counters'][base_name]}"

    @staticmethod
    def clear() -> None:
        """Clear all objects (but preserve undo/redo stacks for recovery)."""
        _session_state["objects"].clear()
        _session_state["current"] = None
        _session_state["history"].clear()
        _session_state["counters"].clear()
        # Note: We don't clear undo_stack/redo_stack so clear can be undone

    @staticmethod
    def get_history(limit: int = 50) -> List[dict]:
        """Get operation history."""
        return _session_state["history"][-limit:]

    @staticmethod
    def get_hierarchy() -> dict:
        """Get object hierarchy."""
        roots = []
        for name, cq_obj in _session_state["objects"].items():
            if cq_obj.parent is None:
                roots.append(StateManager._build_tree(name))
        return {"roots": roots, "current": _session_state["current"]}

    @staticmethod
    def _build_tree(name: str) -> dict:
        """Build tree structure for hierarchy."""
        cq_obj = _session_state["objects"].get(name)
        if cq_obj is None:
            return {"name": name, "children": []}
        return {
            "name": name,
            "obj_type": cq_obj.obj_type,
            "children": [StateManager._build_tree(child) for child in cq_obj.children]
        }

    @staticmethod
    def save_snapshot(description: str) -> None:
        """Save current state to undo stack with deep copy.

        Note: Some complex imported geometry (STEP files) may not be deep-copyable.
        In such cases, the snapshot is skipped and undo/redo will not be available
        for that operation.
        """
        try:
            snapshot = StateSnapshot(
                timestamp=datetime.now(),
                objects=copy.deepcopy(_session_state["objects"]),
                current=_session_state["current"],
                description=description
            )
            _session_state["undo_stack"].append(snapshot)
            _session_state["redo_stack"].clear()  # Clear redo on new operation

            # Limit undo stack size
            if len(_session_state["undo_stack"]) > MAX_UNDO_STACK_SIZE:
                _session_state["undo_stack"].pop(0)
        except Exception:
            # Deep copy failed (likely due to complex imported geometry)
            # Skip snapshot to allow operation to continue
            # Undo/redo will not be available for this operation
            pass

    @staticmethod
    def undo() -> Optional[str]:
        """Undo last operation."""
        if not _session_state["undo_stack"]:
            return None

        # Save current state to redo stack first
        current_snapshot = StateSnapshot(
            timestamp=datetime.now(),
            objects=copy.deepcopy(_session_state["objects"]),
            current=_session_state["current"],
            description="redo_point"
        )

        # Pop and restore previous state
        snapshot = _session_state["undo_stack"].pop()
        _session_state["objects"] = snapshot.objects
        _session_state["current"] = snapshot.current
        _session_state["redo_stack"].append(current_snapshot)

        return snapshot.description

    @staticmethod
    def redo() -> Optional[str]:
        """Redo previously undone operation."""
        if not _session_state["redo_stack"]:
            return None

        # Save current state to undo stack first
        current_snapshot = StateSnapshot(
            timestamp=datetime.now(),
            objects=copy.deepcopy(_session_state["objects"]),
            current=_session_state["current"],
            description="undo_point"
        )

        # Pop and restore next state
        snapshot = _session_state["redo_stack"].pop()
        _session_state["objects"] = snapshot.objects
        _session_state["current"] = snapshot.current
        _session_state["undo_stack"].append(current_snapshot)

        return snapshot.description

    @staticmethod
    def can_undo() -> bool:
        """Check if undo is available."""
        return len(_session_state["undo_stack"]) > 0

    @staticmethod
    def can_redo() -> bool:
        """Check if redo is available."""
        return len(_session_state["redo_stack"]) > 0

    @staticmethod
    def set_current(name: str) -> None:
        """Set the current active object."""
        if name not in _session_state["objects"]:
            raise ValueError(f"Object '{name}' not found")
        _session_state["current"] = name


# =============================================================================
# REFERENCE RESOLUTION
# =============================================================================

def resolve_value(value: Any) -> Any:
    """Recursively resolve references and type constructions."""
    if isinstance(value, dict):
        if "_ref" in value:
            # Reference to stored object
            return StateManager.get(value["_ref"])
        elif "_type" in value:
            # Inline type construction
            return construct_type(value)
        else:
            # Recursively resolve dict values
            return {k: resolve_value(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [resolve_value(v) for v in value]
    return value


def construct_type(spec: dict) -> Any:
    """Construct CadQuery types from specifications."""
    type_name = spec["_type"]

    if type_name == "Vector":
        x = spec.get("x", 0)
        y = spec.get("y", 0)
        z = spec.get("z", 0)
        return Vector(x, y, z)

    elif type_name == "Plane":
        if "name" in spec:
            # Named plane using classmethod
            plane_name = spec["name"]
            return Plane.named(plane_name)
        else:
            # Custom plane
            origin = spec.get("origin", (0, 0, 0))
            xDir = spec.get("xDir", (1, 0, 0))
            normal = spec.get("normal", (0, 0, 1))
            if isinstance(origin, list):
                origin = tuple(origin)
            if isinstance(xDir, list):
                xDir = tuple(xDir)
            if isinstance(normal, list):
                normal = tuple(normal)
            return Plane(origin=origin, xDir=xDir, normal=normal)

    elif type_name == "Location":
        if "plane" in spec:
            plane = resolve_value(spec["plane"])
            if "vector" in spec:
                vec = resolve_value(spec["vector"])
                return Location(plane, vec)
            return Location(plane)
        elif "vector" in spec:
            vec = resolve_value(spec["vector"])
            return Location(vec)
        else:
            x = spec.get("x", 0)
            y = spec.get("y", 0)
            z = spec.get("z", 0)
            rx = spec.get("rx", 0)
            ry = spec.get("ry", 0)
            rz = spec.get("rz", 0)
            # Location supports (x, y, z, rx, ry, rz) constructor
            return Location(x, y, z, rx, ry, rz)

    elif type_name == "Color":
        if "name" in spec:
            return Color(spec["name"])
        else:
            r = spec.get("r", 0)
            g = spec.get("g", 0)
            b = spec.get("b", 0)
            a = spec.get("a", 1)
            return Color(r, g, b, a)

    elif type_name == "Matrix":
        values = spec.get("values", None)
        if values:
            return Matrix(values)
        return Matrix()

    else:
        raise ValueError(f"Unknown type: {type_name}")


def get_object_properties(obj: Any) -> dict:
    """Get properties from a CadQuery object."""
    props = {}

    try:
        # Try to get geometric properties
        if hasattr(obj, 'val'):
            shape = obj.val()
        else:
            shape = obj

        if hasattr(shape, 'Volume'):
            try:
                props["volume"] = shape.Volume()
            except Exception:
                pass  # Shape doesn't support volume property

        if hasattr(shape, 'Area'):
            try:
                props["area"] = shape.Area()
            except Exception:
                pass  # Shape doesn't support area property

        if hasattr(shape, 'Center'):
            try:
                center = shape.Center()
                props["center"] = [center.x, center.y, center.z]
            except Exception:
                pass  # Shape doesn't support center property

        if hasattr(shape, 'BoundingBox'):
            try:
                bb = shape.BoundingBox()
                props["bounding_box"] = {
                    "xmin": bb.xmin, "xmax": bb.xmax,
                    "ymin": bb.ymin, "ymax": bb.ymax,
                    "zmin": bb.zmin, "zmax": bb.zmax
                }
            except Exception:
                pass  # Shape doesn't support bounding box property
    except Exception:
        pass  # Failed to extract properties from object

    return props


def determine_obj_type(obj: Any) -> str:
    """Determine the type of a CadQuery object."""
    type_name = type(obj).__name__

    # Handle Workplane specially
    if isinstance(obj, Workplane):
        return "Workplane"
    elif isinstance(obj, Sketch):
        return "Sketch"
    elif isinstance(obj, Assembly):
        return "Assembly"
    elif isinstance(obj, Solid):
        return "Solid"
    elif isinstance(obj, Compound):
        return "Compound"
    elif isinstance(obj, Face):
        return "Face"
    elif isinstance(obj, Wire):
        return "Wire"
    elif isinstance(obj, Edge):
        return "Edge"
    elif isinstance(obj, Vertex):
        return "Vertex"
    elif isinstance(obj, Shell):
        return "Shell"
    elif isinstance(obj, Vector):
        return "Vector"
    elif isinstance(obj, Plane):
        return "Plane"
    elif isinstance(obj, Location):
        return "Location"
    elif isinstance(obj, Matrix):
        return "Matrix"
    elif isinstance(obj, Selector):
        return "Selector"
    else:
        return type_name


# =============================================================================
# HELPER FUNCTIONS FOR cq_feedback TOOL
# =============================================================================

def _count_edge_types(edges: list) -> dict[str, int]:
    """Count edge types (Line, Arc, Circle, etc.)."""
    from collections import Counter
    types = [e.geomType() for e in edges]
    return dict(Counter(types))


def _count_face_types(faces: list) -> dict[str, int]:
    """Count face types (Plane, Cylinder, Sphere, etc.)."""
    from collections import Counter
    types = [f.geomType() for f in faces]
    return dict(Counter(types))


def _encode_image_base64(image_bytes: bytes) -> str:
    """
    Encode image bytes to base64 string.

    Args:
        image_bytes: Raw image bytes (PNG, JPG, etc.)

    Returns:
        Base64 encoded string
    """
    import base64
    return base64.b64encode(image_bytes).decode('utf-8')


def _rotate_png(png_bytes: bytes, angle: int) -> bytes:
    """
    Rotate a PNG image by specified angle.

    Args:
        png_bytes: PNG image bytes
        angle: Rotation angle in degrees (positive = counter-clockwise)
               90 = rotate left, -90 = rotate right, 180 = upside down

    Returns:
        Rotated PNG image bytes
    """
    from PIL import Image
    import io

    # Load PNG from bytes
    img = Image.open(io.BytesIO(png_bytes))

    # Rotate image (PIL rotates counter-clockwise by default)
    rotated = img.rotate(angle, expand=True)

    # Convert back to PNG bytes
    output = io.BytesIO()
    rotated.save(output, format='PNG')
    return output.getvalue()


def _detect_features(shape: Any, enable_qualitative: bool = True) -> dict[str, Any]:
    """
    Detect common geometric features.

    Provides both quantitative and optional qualitative descriptions.

    Args:
        shape: Shape object to analyze
        enable_qualitative: If True, include human-readable feature descriptions

    Returns:
        Dictionary with quantitative data and optional qualitative descriptions
    """
    result = {
        "quantitative": {},
        "qualitative": []
    }

    faces = shape.Faces()
    edges = shape.Edges()

    # Quantitative: Face type counts
    face_types = _count_face_types(faces)
    result["quantitative"]["face_types"] = face_types

    # Quantitative: Edge type counts
    edge_types = _count_edge_types(edges)
    result["quantitative"]["edge_types"] = edge_types

    # Optional qualitative descriptions
    if enable_qualitative:
        # Interpret face types
        if "PLANE" in face_types and face_types["PLANE"] >= 6:
            result["qualitative"].append("box-like shape with planar faces")
        elif "PLANE" in face_types:
            result["qualitative"].append(f"{face_types['PLANE']} planar faces")

        if "CYLINDER" in face_types:
            result["qualitative"].append(f"{face_types['CYLINDER']} cylindrical features detected")

        if "SPHERE" in face_types:
            result["qualitative"].append(f"{face_types['SPHERE']} spherical surfaces")

        # Interpret edge types
        if "CIRCLE" in edge_types:
            if face_types.get("CYLINDER", 0) > 0:
                result["qualitative"].append(f"{edge_types['CIRCLE']} circular edges (likely cylindrical holes/features)")
            else:
                result["qualitative"].append(f"{edge_types['CIRCLE']} circular edges")

        if "LINE" in edge_types:
            result["qualitative"].append(f"{edge_types['LINE']} straight edges")

        # Detect common patterns
        total_faces = len(faces)
        if total_faces == 6 and face_types.get("PLANE", 0) == 6:
            result["qualitative"].append("simple box/cuboid geometry")
        elif "CYLINDER" in face_types and "PLANE" in face_types:
            result["qualitative"].append("mixed planar and cylindrical geometry (possibly holes or bosses)")

    return result


def _describe_shape(shape: Any, detail_level: str, enable_qualitative: bool = True) -> dict[str, Any]:
    """Describe a single Shape object."""

    result = {
        "type": "Shape",
        "shape_type": shape.ShapeType(),
        "summary": f"{shape.ShapeType()} object"
    }

    # Basic topology
    result["topology"] = {
        "faces": len(shape.Faces()),
        "edges": len(shape.Edges()),
        "vertices": len(shape.Vertices())
    }

    # Geometric properties
    bbox = shape.BoundingBox()
    result["bounding_box"] = {
        "dimensions": {
            "x": bbox.xlen,
            "y": bbox.ylen,
            "z": bbox.zlen
        },
        "center": bbox.center.toTuple(),
        "min": (bbox.xmin, bbox.ymin, bbox.zmin),
        "max": (bbox.xmax, bbox.ymax, bbox.zmax)
    }

    # Properties (only for solids)
    if shape.ShapeType() == "Solid":
        result["properties"] = {
            "volume": shape.Volume(),
            "surface_area": shape.Area(),
            "center_of_mass": shape.Center().toTuple()
        }
    else:
        result["properties"] = {
            "area": shape.Area() if hasattr(shape, 'Area') else None
        }

    # Standard detail level stops here
    if detail_level == "basic":
        return result

    # Add edge analysis
    edges = shape.Edges()
    result["edge_analysis"] = {
        "count": len(edges),
        "total_length": sum(e.Length() for e in edges),
        "types": _count_edge_types(edges)
    }

    # Add face analysis
    faces = shape.Faces()
    result["face_analysis"] = {
        "count": len(faces),
        "types": _count_face_types(faces),
        "areas": [f.Area() for f in faces] if detail_level == "comprehensive" else None
    }

    # Comprehensive detail level
    if detail_level == "comprehensive":
        # Vertex positions
        vertices = shape.Vertices()
        result["vertices"] = [v.toTuple() for v in vertices]

        # Feature detection (pass through qualitative parameter)
        result["features"] = _detect_features(shape, enable_qualitative=enable_qualitative)

    return result


def _describe_assembly(assy: Assembly, detail_level: str, enable_qualitative: bool = True) -> dict[str, Any]:
    """Describe Assembly with part hierarchy."""

    result = {
        "type": "Assembly",
        "name": assy.name,
        "location": assy.loc.toTuple(),
        "metadata": assy.metadata
    }

    # Count children
    children = list(assy.children)
    result["structure"] = {
        "num_children": len(children),
        "parts": []
    }

    # Traverse assembly tree
    for name, child in assy.traverse():
        part_info = {
            "name": name,
            "location": child.loc.toTuple()
        }

        # Analyze shape if present
        if child.obj:
            part_info["shape"] = _describe_shape(child.obj, "basic", enable_qualitative=False)

        result["structure"]["parts"].append(part_info)

    # Comprehensive detail
    if detail_level == "comprehensive":
        # Compute overall bounding box
        compound = assy.toCompound()
        bbox = compound.BoundingBox()
        result["overall_bounds"] = {
            "dimensions": {
                "x": bbox.xlen,
                "y": bbox.ylen,
                "z": bbox.zlen
            },
            "center": bbox.center.toTuple()
        }

        # Total volume
        result["total_volume"] = compound.Volume()

    return result


def _generate_detailed_description(
    obj: Any,
    detail_level: str = "step_text",
    enable_qualitative: bool = False
) -> dict[str, Any]:
    """
    Generate comprehensive text description of CadQuery object.

    Args:
        obj: Workplane, Sketch, Assembly, or Shape
        detail_level: "basic", "standard", "comprehensive", or "step_text"
        enable_qualitative: Enable qualitative feature descriptions
            (only applies when detail_level="comprehensive")

    Returns:
        Structured description dictionary
    """

    # Handle STEP text export
    if detail_level == "step_text":
        temp_dir = tempfile.gettempdir()
        step_path = os.path.join(temp_dir, "cadquery_temp_export.step")

        try:
            # Export based on object type
            if isinstance(obj, Assembly):
                obj.save(step_path)
            elif isinstance(obj, Sketch):
                # Convert sketch to workplane for export
                wp = Workplane().placeSketch(obj)
                exporters.export(wp, step_path, exportType=exporters.ExportTypes.STEP)
            else:
                # Workplane or Shape
                exporters.export(obj, step_path, exportType=exporters.ExportTypes.STEP)

            # Read STEP file content
            with open(step_path, 'r', encoding='utf-8') as f:
                step_content = f.read()

            return {
                "type": "STEP Export",
                "format": "STEP (ISO 10303-21)",
                "content": step_content,
                "content_length": len(step_content)
            }
        finally:
            # Always clean up temp file
            if os.path.exists(step_path):
                try:
                    os.remove(step_path)
                except:
                    pass  # Ignore cleanup errors

    # Handle different object types
    if isinstance(obj, Assembly):
        return _describe_assembly(obj, detail_level, enable_qualitative)
    elif isinstance(obj, Workplane):
        stack_objects = obj.vals()
        if len(stack_objects) == 1:
            return _describe_shape(stack_objects[0], detail_level, enable_qualitative)
        else:
            # Multiple objects on stack
            return {
                "type": "Workplane with multiple objects",
                "count": len(stack_objects),
                "objects": [_describe_shape(o, detail_level, enable_qualitative) for o in stack_objects]
            }
    elif isinstance(obj, Sketch):
        # Basic sketch description
        return {
            "type": "Sketch",
            "summary": "2D Sketch object"
        }
    else:
        # Assume it's a Shape
        return _describe_shape(obj, detail_level, enable_qualitative)


async def _svg_to_png_playwright(svg_content: str, width: int = 400, height: int = 300) -> bytes:
    """
    Convert SVG to PNG using Playwright headless browser (Windows compatible).

    Args:
        svg_content: SVG XML string
        width: Output PNG width in pixels
        height: Output PNG height in pixels

    Returns:
        PNG image bytes

    Raises:
        Exception: If conversion fails
    """
    from playwright.async_api import async_playwright
    from PIL import Image
    import io
    import base64

    async with async_playwright() as p:
        # Launch browser (tries Chromium first)
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={'width': width, 'height': height})

        try:
            # Create data URI for SVG
            svg_b64 = base64.b64encode(svg_content.encode('utf-8')).decode('ascii')
            data_uri = f"data:image/svg+xml;base64,{svg_b64}"

            # Wrap SVG in HTML with object-fit:contain for proper scaling without cropping
            html_content = f"""<!DOCTYPE html>
<html>
<head>
<style>
* {{ margin: 0; padding: 0; }}
body {{
    margin: 0;
    padding: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    width: {width}px;
    height: {height}px;
    background: #ffffff;
    overflow: hidden;
}}
img {{
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
}}
</style>
</head>
<body>
<img src="{data_uri}" alt="CAD Model"/>
</body>
</html>"""

            # Navigate to HTML content
            await page.set_content(html_content, wait_until='networkidle', timeout=10000)

            # Take screenshot
            png_bytes = await page.screenshot(type='png', full_page=False)

            return png_bytes

        finally:
            await browser.close()


def _svg_to_png_fallback(svg_content: str, width: int = 400, height: int = 300) -> bytes:
    """
    Fallback PNG conversion using svglib + Pillow (pure Python).

    Args:
        svg_content: SVG XML string
        width: Output PNG width
        height: Output PNG height

    Returns:
        PNG image bytes

    Raises:
        Exception: If conversion fails
    """
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPM
    from PIL import Image
    import io
    import tempfile

    # svglib requires file path
    temp_dir = tempfile.gettempdir()
    svg_path = os.path.join(temp_dir, "temp_conversion.svg")

    try:
        with open(svg_path, 'w', encoding='utf-8') as f:
            f.write(svg_content)

        # Convert SVG to ReportLab drawing
        drawing = svg2rlg(svg_path)

        if not drawing:
            raise Exception("Failed to parse SVG with svglib")

        # Render to PNG bytes
        png_data = renderPM.drawToString(drawing, fmt='PNG')

        # Scale to fit within target dimensions while preserving aspect ratio
        img = Image.open(io.BytesIO(png_data))
        orig_width, orig_height = img.size

        # Calculate scale factor to fit within target dimensions
        scale_w = width / orig_width
        scale_h = height / orig_height
        scale = min(scale_w, scale_h)  # Use the smaller scale to ensure it fits

        # Calculate new dimensions
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)

        # Resize image with aspect ratio preserved
        img_resized = img.resize((new_width, new_height), Image.LANCZOS)

        # Create new image with target dimensions and white background
        final_img = Image.new('RGB', (width, height), 'white')

        # Calculate position to center the resized image
        paste_x = (width - new_width) // 2
        paste_y = (height - new_height) // 2

        # Paste resized image onto white background
        if img_resized.mode == 'RGBA':
            final_img.paste(img_resized, (paste_x, paste_y), img_resized)
        else:
            final_img.paste(img_resized, (paste_x, paste_y))

        # Convert to PNG bytes
        output = io.BytesIO()
        final_img.save(output, format='PNG')
        return output.getvalue()

    finally:
        # Clean up temp file
        if os.path.exists(svg_path):
            try:
                os.remove(svg_path)
            except:
                pass


async def _svg_to_png(svg_content: str, width: int = 400, height: int = 300) -> bytes:
    """
    Convert SVG to PNG using best available method.

    Tries Playwright first (Windows compatible), falls back to svglib if unavailable.

    Args:
        svg_content: SVG XML string
        width: Output PNG width in pixels
        height: Output PNG height in pixels

    Returns:
        PNG image bytes

    Raises:
        Exception: If all conversion methods fail
    """
    # Try playwright (primary - works on Windows)
    if _PNG_LIBRARY == "playwright":
        try:
            return await _svg_to_png_playwright(svg_content, width, height)
        except Exception as e:
            # If playwright fails, try fallback
            try:
                return _svg_to_png_fallback(svg_content, width, height)
            except Exception as fallback_err:
                raise Exception(
                    f"PNG conversion failed. Playwright error: {str(e)}. "
                    f"Fallback error: {str(fallback_err)}"
                )

    # Try svglib fallback (pure Python)
    elif _PNG_LIBRARY == "svglib+pillow":
        return _svg_to_png_fallback(svg_content, width, height)

    else:
        raise Exception(
            "PNG conversion not available. "
            "Install playwright: pip install playwright pillow && playwright install chromium"
        )


async def _export_svg_preview(
    obj: Workplane | Sketch | Assembly,
    width: int = 400,
    height: int = 300,
    projection_dir: tuple[float, float, float] = (3, -2, 1),
    show_axes: bool = False,
    show_hidden: bool = False,
    stroke_color: tuple[int, int, int] = (0, 0, 0),
    hidden_color: tuple[int, int, int] = (160, 160, 160),
    stroke_width: float = -1.0,  # Auto-calculate
    focus: float | None = None,  # Perspective projection
    format: str = "png"  # Output format: "svg" or "png" (DEFAULT: PNG per user requirement)
) -> str | bytes:
    """
    Export object to SVG or PNG for visual feedback.

    Args:
        obj: CadQuery object to export
        width, height: Image dimensions in pixels
        projection_dir: Camera direction (x, y, z) tuple for orthographic projection (default: isometric front right)
        show_axes: Display coordinate axes indicator
        show_hidden: Show hidden edges as dashed lines (DEFAULT: False)
        stroke_color: RGB tuple (0-255) for visible edge color
        hidden_color: RGB tuple (0-255) for hidden edge color
        stroke_width: Line thickness (-1.0 = auto-calculate)
        focus: Distance for perspective projection (None = orthographic)
        format: Output format - "svg" (string) or "png" (bytes). DEFAULT: "png"

    Returns:
        SVG content as string OR PNG content as bytes (depending on format parameter)
    """
    try:
        # Create temp file
        temp_dir = tempfile.gettempdir()
        svg_path = os.path.join(temp_dir, "cadquery_preview.svg")

        # Build SVG export options
        opt = {
            "width": width,
            "height": height,
            "marginLeft": 5,
            "marginRight": 5,
            "marginTop": 5,
            "marginBottom": 5,
            "showAxes": show_axes,
            "projectionDir": projection_dir,
            "showHidden": show_hidden,
            "strokeColor": stroke_color,
            "hiddenColor": hidden_color,
            "strokeWidth": stroke_width
        }

        # Add perspective projection if specified
        if focus is not None:
            opt["focus"] = focus

        # Export based on type
        if isinstance(obj, Assembly):
            # Use toCompound() to preserve assembly locations
            compound = obj.toCompound()
            export_obj = Workplane().add(compound)
            exporters.export(export_obj, svg_path,
                           exportType=exporters.ExportTypes.SVG,
                           opt=opt)
        elif isinstance(obj, Sketch):
            # Convert sketch to face for export
            wp = Workplane().placeSketch(obj)
            exporters.export(wp, svg_path,
                           exportType=exporters.ExportTypes.SVG,
                           opt=opt)
        else:
            # Workplane or Shape
            exporters.export(obj, svg_path,
                           exportType=exporters.ExportTypes.SVG,
                           opt=opt)

        # Read SVG content
        if not os.path.exists(svg_path):
            raise Exception("SVG export failed - file not created")

        with open(svg_path, 'r') as f:
            svg_content = f.read()

        # NEW: Convert to PNG if requested (DEFAULT)
        if format.lower() == "png":
            try:
                # Now that this function is async, we can await directly
                png_bytes = await _svg_to_png(svg_content, width, height)
                return png_bytes
            except Exception as e:
                raise Exception(
                    f"PNG conversion failed: {str(e)}. "
                    f"Install playwright: pip install playwright pillow && playwright install chromium"
                )

        # Return SVG only if explicitly requested
        return svg_content

    except Exception as e:
        return f"Export error: {str(e)}"


async def _export_multiple_views(
    obj: Workplane | Sketch | Assembly,
    views: list[str] | list[tuple[float, float, float]],
    width: int = 400,
    height: int = 300,
    show_hidden: bool = False,
    show_axes: bool = False,
    format: str = "png",  # Output format: "svg" or "png" (DEFAULT: PNG per user requirement)
    **svg_options
) -> dict[str, dict[str, Any]]:
    """
    Export object from multiple viewing angles.

    Args:
        obj: CadQuery object to render
        views: List of view names (e.g., ["front", "top"]) OR
               list of projection direction tuples
        width, height: Image dimensions in pixels
        show_hidden: Show hidden lines as dashed (DEFAULT: False)
        show_axes: Display coordinate axes
        format: Output format - "svg" or "png". DEFAULT: "png"
        **svg_options: Additional exporter options (stroke_color, focus, etc.)

    Returns:
        Dictionary mapping view names to view data with human-readable labels
    """
    results = {}

    # Helper to generate human-readable label from view name
    def _generate_label(view_name: str) -> str:
        """Convert view_name to human-readable label."""
        return view_name.replace("_", " ").title()

    for i, view in enumerate(views):
        if isinstance(view, str):
            # Named view
            if view not in STANDARD_VIEWS:
                raise ValueError(f"Unknown view name: {view}. Available views: {list(STANDARD_VIEWS.keys())}")
            proj_dir = STANDARD_VIEWS[view]
            view_name = view
            label = _generate_label(view_name)  # E.g., "isometric_front_right" → "Isometric Front Right"
        else:
            # Custom projection tuple
            proj_dir = view
            view_name = f"custom_{i}"
            label = f"Custom View {i+1}"

        try:
            image_content = await _export_svg_preview(
                obj, width, height, proj_dir,
                show_axes=show_axes,
                show_hidden=show_hidden,
                format=format,  # Pass format parameter
                **svg_options
            )

            # Build view data based on format
            view_data = {
                "projection": proj_dir,
                "view_name": view_name,
                "label": label,  # HUMAN-READABLE LABEL
                "width": width,
                "height": height,
                "format": format
            }

            if format.lower() == "png" and isinstance(image_content, bytes):
                # Apply rotation for specific orthographic views
                if view_name == "front":
                    image_content = _rotate_png(image_content, -90)  # Rotate right (clockwise)
                elif view_name == "back":
                    image_content = _rotate_png(image_content, 90)   # Rotate left (counter-clockwise)
                elif view_name == "right":
                    image_content = _rotate_png(image_content, 90)   # Rotate left (counter-clockwise)
                elif view_name == "left":
                    image_content = _rotate_png(image_content, -90)  # Rotate right (clockwise)
                elif view_name == "bottom":
                    image_content = _rotate_png(image_content, 180)  # Rotate 180 degrees (upside down correction)

                # PNG format (DEFAULT - client requirement)
                view_data["png_bytes"] = image_content  # Store raw bytes for Image class
                # Do NOT include "svg" key per user requirement
            else:
                # SVG format (only if explicitly requested)
                view_data["svg"] = image_content

            results[view_name] = view_data

        except Exception as e:
            results[view_name] = {
                "error": str(e),
                "projection": proj_dir,
                "view_name": view_name,
                "label": label,
                "format": format
            }

    return results


# =============================================================================
# RESOURCE LOADING FOR TOOLS
# =============================================================================
# This function allows tools to load documentation resources without using
# ctx.read_resource(), which is designed for cross-server communication.
# Instead, we directly call load_doc_file() since the resources are on the
# same server.
# =============================================================================

# This will be set after resource handlers are defined
DOCS_DIR = r"C:\Users\user\Desktop\mcp-dev\cadquery\docs_1"


def load_doc_file(filepath: str) -> str:
    """Load a markdown documentation file and return its contents."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return f"Documentation file not found: {filepath}\n\nPlease run generate_docs.py first to create the documentation files."
    except Exception as e:
        return f"Error loading documentation: {str(e)}"


def get_resource_uri_to_file_mapping():
    """Returns a mapping of resource URIs to their file paths.

    This function uses lazy initialization to avoid forward reference issues.
    """
    return {
        # Fluent API
        "cq://docs/fluent-api/Workplane": os.path.join(DOCS_DIR, "fluent-api", "Workplane.md"),
        "cq://docs/fluent-api/Assembly": os.path.join(DOCS_DIR, "fluent-api", "Assembly.md"),
        "cq://docs/fluent-api/Sketch": os.path.join(DOCS_DIR, "fluent-api", "Sketch.md"),
        # Direct API
        "cq://docs/direct-api/Shape": os.path.join(DOCS_DIR, "direct-api", "Shape.md"),
        "cq://docs/direct-api/Vertex": os.path.join(DOCS_DIR, "direct-api", "Vertex.md"),
        "cq://docs/direct-api/Edge": os.path.join(DOCS_DIR, "direct-api", "Edge.md"),
        "cq://docs/direct-api/Wire": os.path.join(DOCS_DIR, "direct-api", "Wire.md"),
        "cq://docs/direct-api/Face": os.path.join(DOCS_DIR, "direct-api", "Face.md"),
        "cq://docs/direct-api/Shell": os.path.join(DOCS_DIR, "direct-api", "Shell.md"),
        "cq://docs/direct-api/Solid": os.path.join(DOCS_DIR, "direct-api", "Solid.md"),
        "cq://docs/direct-api/CompSolid": os.path.join(DOCS_DIR, "direct-api", "CompSolid.md"),
        "cq://docs/direct-api/Compound": os.path.join(DOCS_DIR, "direct-api", "Compound.md"),
        "cq://docs/direct-api/Mixin1D": os.path.join(DOCS_DIR, "direct-api", "Mixin1D.md"),
        "cq://docs/direct-api/Mixin3D": os.path.join(DOCS_DIR, "direct-api", "Mixin3D.md"),
        # Geometry API
        "cq://docs/geometry-api/Vector": os.path.join(DOCS_DIR, "geometry-api", "Vector.md"),
        "cq://docs/geometry-api/Matrix": os.path.join(DOCS_DIR, "geometry-api", "Matrix.md"),
        "cq://docs/geometry-api/Plane": os.path.join(DOCS_DIR, "geometry-api", "Plane.md"),
        "cq://docs/geometry-api/BoundBox": os.path.join(DOCS_DIR, "geometry-api", "BoundBox.md"),
        "cq://docs/geometry-api/Location": os.path.join(DOCS_DIR, "geometry-api", "Location.md"),
        # Selectors API
        "cq://docs/selectors-api/Selector": os.path.join(DOCS_DIR, "selectors-api", "Selector.md"),
        "cq://docs/selectors-api/NearestToPointSelector": os.path.join(DOCS_DIR, "selectors-api", "NearestToPointSelector.md"),
        "cq://docs/selectors-api/BoxSelector": os.path.join(DOCS_DIR, "selectors-api", "BoxSelector.md"),
        "cq://docs/selectors-api/BaseDirSelector": os.path.join(DOCS_DIR, "selectors-api", "BaseDirSelector.md"),
        "cq://docs/selectors-api/ParallelDirSelector": os.path.join(DOCS_DIR, "selectors-api", "ParallelDirSelector.md"),
        "cq://docs/selectors-api/DirectionSelector": os.path.join(DOCS_DIR, "selectors-api", "DirectionSelector.md"),
        "cq://docs/selectors-api/PerpendicularDirSelector": os.path.join(DOCS_DIR, "selectors-api", "PerpendicularDirSelector.md"),
        "cq://docs/selectors-api/TypeSelector": os.path.join(DOCS_DIR, "selectors-api", "TypeSelector.md"),
        "cq://docs/selectors-api/RadiusNthSelector": os.path.join(DOCS_DIR, "selectors-api", "RadiusNthSelector.md"),
        "cq://docs/selectors-api/CenterNthSelector": os.path.join(DOCS_DIR, "selectors-api", "CenterNthSelector.md"),
        "cq://docs/selectors-api/DirectionMinMaxSelector": os.path.join(DOCS_DIR, "selectors-api", "DirectionMinMaxSelector.md"),
        "cq://docs/selectors-api/DirectionNthSelector": os.path.join(DOCS_DIR, "selectors-api", "DirectionNthSelector.md"),
        "cq://docs/selectors-api/LengthNthSelector": os.path.join(DOCS_DIR, "selectors-api", "LengthNthSelector.md"),
        "cq://docs/selectors-api/AreaNthSelector": os.path.join(DOCS_DIR, "selectors-api", "AreaNthSelector.md"),
        "cq://docs/selectors-api/BinarySelector": os.path.join(DOCS_DIR, "selectors-api", "BinarySelector.md"),
        "cq://docs/selectors-api/AndSelector": os.path.join(DOCS_DIR, "selectors-api", "AndSelector.md"),
        "cq://docs/selectors-api/SumSelector": os.path.join(DOCS_DIR, "selectors-api", "SumSelector.md"),
        "cq://docs/selectors-api/SubtractSelector": os.path.join(DOCS_DIR, "selectors-api", "SubtractSelector.md"),
        "cq://docs/selectors-api/InverseSelector": os.path.join(DOCS_DIR, "selectors-api", "InverseSelector.md"),
        "cq://docs/selectors-api/StringSyntaxSelector": os.path.join(DOCS_DIR, "selectors-api", "StringSyntaxSelector.md"),
        # Free Functions
        "cq://docs/free-functions/all": os.path.join(DOCS_DIR, "free-functions", "all.md"),
        # Overview
        "cq://docs/overview": os.path.join(DOCS_DIR, "overview.md"),
    }


def load_relevant_docs_direct(resource_uris: List[str]) -> str:
    """Load documentation by directly reading files instead of using ctx.read_resource().

    This is the correct approach for tools accessing resources on the same server.
    ctx.read_resource() is designed for client-to-server communication, not for
    tools reading resources from their own server.

    Args:
        resource_uris: List of resource URIs to load

    Returns:
        Combined markdown documentation as a single string
    """
    uri_to_file = get_resource_uri_to_file_mapping()
    docs = []

    for uri in resource_uris:
        file_path = uri_to_file.get(uri)
        if file_path:
            try:
                content = load_doc_file(file_path)
                if content:
                    docs.append(f"### Documentation: {uri}\n{content}")
            except Exception as e:
                # Log but don't fail completely
                docs.append(f"### Documentation: {uri}\nError loading: {str(e)}")
        else:
            docs.append(f"### Documentation: {uri}\nResource URI not found in mapping")

    return "\n\n".join(docs) if docs else ""




# =============================================================================
# MCP TOOLS
# =============================================================================
# Note: All MCP tools must be async per FastMCP requirements, even though
# they don't use await internally. This is a FastMCP framework requirement.

@mcp.tool(name="cq_fluent_api")
async def cq_fluent_api(
    class_name: Literal["Workplane", "Sketch", "Assembly"],
    operations: List[dict],
    start_from: Optional[str] = None,
    store_as: Optional[str] = None,
    init_params: Optional[dict] = None
) -> str:
    """
    Execute chainable operations on Workplane, Sketch, or Assembly objects.
    This is the primary tool for high-level 3D modeling using CadQuery's fluent API.

    IMPORTANT: Before using this tool, you MUST first call the cq_load_docs tool to load the relevant documentation for this API tool !!! 

    Args:
        class_name: The CadQuery class to use
            - "Workplane": 2D/3D modeling with method chaining
            - "Sketch": 2D constraint-based sketching
            - "Assembly": Hierarchical assembly with constraints
        operations: List of operations to chain. Each operation is a dict:
            - method (str): Method name to call (e.g., "box", "fillet", "faces")
            - params (dict): Parameters for the method (e.g., {"length": 10, "width": 20})
        start_from: Optional name of existing object to continue from.
            If provided, chains operations starting from this stored object.
        store_as: Name to store the result. Auto-generated if not provided.
        init_params: Optional initialization parameters for the class.
            Examples:
            - Workplane with named plane:
              {"plane": "XY"}  # or "YZ", "XZ"

            - Workplane with custom Plane object:
              {"plane": {"_type": "Plane", "origin": [0,0,10], "normal": [0,0,1]}}  # IMPORTANT (Remember this whenever using the custom Plane object): valid xDir and normal values for each side of the workplane: front side: {"xDir": [1,0,0], "normal": [0,-1,0]}, back side: {"xDir": [-1,0,0], "normal": [0,1,0]}, right side: {"xDir": [0,1,0], "normal": [1,0,0]}, left side: {"xDir": [0,-1,0], "normal": [-1,0,0]}, top side: {"xDir": [1,0,0], "normal": [0,0,1]}, bottom side: {"xDir": [1,0,0], "normal": [0,0,-1]}

            - Sketch (usually no special params):
              {}  # or omit init_params entirely

            - Assembly with name and color:
              {"name": "my_assembly", "color": {"_type": "Color", "name": "red"}}
              {"name": "my_assembly", "color": {"_type": "Color", "r": 1, "g": 0, "b": 0, "a": 1}}

    Returns:
        JSON string with: status, name, obj_type, properties (volume, area, center, bounding_box)

    Workplane Methods:
        Primitives: box, sphere, cylinder, cone, torus, wedge
        2D Shapes: circle, rect, polygon, polyline, spline, ellipse, slot2D, text
        3D Operations: extrude, revolve, sweep, loft, cutThruAll, cutBlind, twistExtrude
        Selection: faces, edges, vertices, wires, solids, shells
        Modification: fillet, chamfer, shell, mirror, offset2D, clean
        Boolean: union, cut, intersect, combine
        Positioning: translate, rotate, transformed, center, workplane, moveTo, move
        Lines: line, lineTo, hLine, vLine, polarLine, sagittaArc, radiusArc, tangentArcPoint

    Sketch Methods:
        Shapes: rect, circle, ellipse, trapezoid, slot, regularPolygon, polygon
        Arcs: arc, segment
        Constraints: constrain
        Selection: faces, edges, vertices, wires

    Assembly Methods:
        Building: add, constrain, solve, remove, toCompound, traverse
        Constraint Types: Plane, Point, Axis, PointInPlane, Fixed, FixedPoint, FixedAxis, PointOnLine, FixedRotation

    Error Handling:
        Returns JSON with status="error" for:
        - Unknown class_name (not Workplane/Sketch/Assembly)
        - Referenced start_from object not found
        - Method not found on the target class
        - Invalid operation parameters
        - CadQuery geometry operation failures
    """
    try:
        # Save state for undo
        StateManager.save_snapshot(f"fluent_api({class_name})")

        # Initialize or get starting object
        if start_from:
            obj = StateManager.get(start_from)
            parent = start_from
        else:
            parent = None
            if init_params:
                resolved_init = resolve_value(init_params)
            else:
                resolved_init = {}

            if class_name == "Workplane":
                plane = resolved_init.get("plane", "XY")
                # Workplane accepts both string plane names and Plane objects
                if isinstance(plane, str):
                    obj = Workplane(plane)
                elif isinstance(plane, Plane):
                    # For Plane objects, create default workplane then set the plane
                    obj = Workplane()
                    obj.plane = plane
                else:
                    # Try passing directly (may be a string or valid plane spec)
                    obj = Workplane(plane)
            elif class_name == "Sketch":
                obj = Sketch()
            elif class_name == "Assembly":
                name = resolved_init.get("name", None)
                color = resolved_init.get("color", None)
                if color:
                    color = resolve_value(color)
                obj = Assembly(name=name, color=color)
            else:
                return error_response(f"Unknown class: {class_name}")

        # Apply operations
        for op in operations:
            method_name = op.get("method")
            params = op.get("params", {})

            # Resolve any references in params
            resolved_params = resolve_value(params)

            # Get the method
            if not hasattr(obj, method_name):
                return error_response(f"Method '{method_name}' not found on {type(obj).__name__}")

            method = getattr(obj, method_name)

            # Call the method
            if resolved_params:
                obj = method(**resolved_params)
            else:
                obj = method()

        # Generate name if not provided
        if store_as is None:
            store_as = StateManager.auto_name(class_name.lower())

        # Determine object type
        obj_type = determine_obj_type(obj)

        # Store the result
        StateManager.store(
            name=store_as,
            obj=obj,
            obj_type=obj_type,
            parent=parent,
            operation=f"fluent_api({class_name})",
            parameters={"operations": [op.get("method") for op in operations]}
        )

        show_safe(obj)

        # Get properties
        props = get_object_properties(obj)

        return json.dumps({
            "status": "success",
            "name": store_as,
            "obj_type": obj_type,
            "properties": props
        })

    except Exception as e:
        return error_response(str(e), traceback.format_exc())


@mcp.tool(name="cq_direct_api")
async def cq_direct_api(
    class_name: Literal["Shape", "Vertex", "Edge", "Wire", "Face", "Shell", "Solid", "Compound", "CompSolid"],
    method: str,
    method_params: dict,
    store_as: Optional[str] = None
) -> str:
    """
    Create geometry using direct static constructors for precise control.
    Use this for low-level geometry construction when you need exact control
    over edges, wires, faces, and solids.

    IMPORTANT: Before using this tool, you MUST first call the cq_load_docs tool to load the relevant documentation for this API tool !!! 

    Args:
        class_name: The shape class to use
            - "Shape": Abstract base (use for instance methods on any shape)
            - "Vertex": Single point in space
            - "Edge": Trimmed curve (has Mixin1D methods)
            - "Wire": Series of connected edges (has Mixin1D methods)
            - "Face": Bounded surface
            - "Shell": Outer boundary of surface
            - "Solid": Single solid (has Mixin3D methods)
            - "Compound": Collection of solids (has Mixin3D methods)
            - "CompSolid": Single compsolid (has Mixin3D methods)
        method: Static method name or instance method name
            - Static methods: "makeCircle", "makeBox", "makeFromWires", etc.
            - Instance methods: "fillet", "chamfer", "fuse", "cut", etc.
        method_params: Parameters for the method. Can include:
            - Primitive values: numbers, strings, booleans
            - References: {"_ref": "object_name"} to reference stored objects
            - Type constructions: {"_type": "Vector", "x": 1, "y": 2, "z": 3}
        store_as: Name to store the result. Auto-generated if not provided.

    Returns:
        JSON string with: status, name, obj_type, properties

    Edge Static Methods:
        makeLine, makeCircle, makeEllipse, makeSpline, makeSplineApprox,
        makeThreePointArc, makeTangentArc, makeBezier

    Wire Static Methods:
        makeCircle, makeEllipse, makePolygon, makeHelix, combine, assembleEdges

    Face Static Methods:
        makeFromWires, makeRuledSurface, makeSplineApprox, makePlane

    Solid Static Methods:
        makeBox, makeCylinder, makeCone, makeSphere, makeTorus, makeWedge,
        makeLoft, sweep, revolve, extrudeLinear

    Compound Static Methods:
        makeCompound, makeText

    Shape Instance Methods (all classes):
        Center, BoundingBox, Area, Volume, isValid, isNull,
        moved, move, located, rotate, rotated, mirror, mirrored,
        translate, transformed, scale, fuse, cut, intersect,
        Faces, Edges, Vertices, Wires, Shells, Solids, Compounds

    Error Handling:
        Returns JSON with status="error" for:
        - Unknown class_name
        - Static method not found on class
        - Instance method not found (when using "self" param)
        - Invalid method parameters
        - CadQuery construction failures
    """
    try:
        # Save state for undo
        StateManager.save_snapshot(f"direct_api({class_name}.{method})")

        # Resolve parameters
        resolved_params = resolve_value(method_params)

        # Get the class
        class_map = {
            "Shape": cq.Shape if hasattr(cq, 'Shape') else None,
            "Vertex": Vertex,
            "Edge": Edge,
            "Wire": Wire,
            "Face": Face,
            "Shell": Shell,
            "Solid": Solid,
            "Compound": Compound,
            "CompSolid": cq.CompSolid if hasattr(cq, 'CompSolid') else Compound
        }

        cls = class_map.get(class_name)
        if cls is None:
            return error_response(f"Unknown class: {class_name}")

        # Check if this is an instance method call (has "self" in params)
        if "self" in resolved_params:
            instance = resolved_params.pop("self")
            if not hasattr(instance, method):
                return error_response(f"Method '{method}' not found on {type(instance).__name__}")
            method_func = getattr(instance, method)
            result = method_func(**resolved_params) if resolved_params else method_func()
        else:
            # Static method call
            if not hasattr(cls, method):
                return error_response(f"Static method '{method}' not found on {class_name}")
            method_func = getattr(cls, method)
            result = method_func(**resolved_params) if resolved_params else method_func()

        # Generate name if not provided
        if store_as is None:
            store_as = StateManager.auto_name(class_name.lower())

        # Determine object type
        obj_type = determine_obj_type(result)

        # Store the result
        StateManager.store(
            name=store_as,
            obj=result,
            obj_type=obj_type,
            operation=f"direct_api({class_name}.{method})",
            parameters={"method": method}
        )

        show_safe(result)

        # Get properties
        props = get_object_properties(result)

        return json.dumps({
            "status": "success",
            "name": store_as,
            "obj_type": obj_type,
            "properties": props
        })

    except Exception as e:
        return error_response(str(e), traceback.format_exc())


@mcp.tool(name="cq_geometry_api")
async def cq_geometry_api(
    class_name: Literal["Vector", "Plane", "Location", "Matrix", "BoundBox"],
    init_params: dict,
    method: Optional[str] = None,
    method_params: Optional[dict] = None,
    store_as: Optional[str] = None
) -> str:
    """
    Create geometry primitives for positioning and transformations.
    Use this to create Vectors, Planes, Locations, Matrices and BoundBoxes
    that can be referenced by other operations.

    IMPORTANT: Before using this tool, you MUST first call the cq_load_docs tool to load the relevant documentation for this API tool !!! 

    Args:
        class_name: The geometry class to create
            - "Vector": 3D vector for positions and directions
            - "Plane": 2D coordinate system in 3D space
            - "Location": Position and orientation (for assembly positioning)
            - "Matrix": 4x4 transformation matrix
            - "BoundBox": Bounding box calculations
        init_params: Initialization parameters (varies by class)
            - Vector: {"x": float, "y": float, "z": float} or {"tuple": [x, y, z]}
            - Plane: {"name": "XY"} or {"origin": [x,y,z], "xDir": [x,y,z], "normal": [x,y,z]}
              Valid plane names: "XY", "YZ", "XZ"
              IMPORTANT (Remember this whenever using the Plane class): valid xDir and normal values for each side of the workplane: front side: {"xDir": [1,0,0], "normal": [0,-1,0]}, back side: {"xDir": [-1,0,0], "normal": [0,1,0]}, right side: {"xDir": [0,1,0], "normal": [1,0,0]}, left side: {"xDir": [0,-1,0], "normal": [-1,0,0]}, top side: {"xDir": [1,0,0], "normal": [0,0,1]}, bottom side: {"xDir": [1,0,0], "normal": [0,0,-1]}
            - Location: Multiple construction patterns:
              {"x": 0, "y": 0, "z": 10}  # Translation only
              {"x": 0, "y": 0, "z": 10, "rx": 0, "ry": 0, "rz": 45}  # Translation + rotation (degrees)
              {"vector": {"_type": "Vector", "x": 0, "y": 0, "z": 10}}  # From Vector
              {"plane": {"_type": "Plane", "name": "XY"}}  # From named Plane
            - Matrix: {"values": [[16 floats]]} or computed from rotations
            - BoundBox: {"shape": {"_ref": "object_name"}} to compute from shape
        method: Optional method to call on the created object
        method_params: Parameters for the method call
        store_as: Name to store the result. Auto-generated if not provided.

    Returns:
        JSON string with: status, name, obj_type, values (class-specific properties)

    Error Handling:
        Returns JSON with status="error" for:
        - Unknown class_name
        - BoundBox without required "shape" parameter
        - Invalid initialization parameters
        - Method not found (when method param specified)
        - Type construction failures
    """
    try:
        # Save state for undo
        StateManager.save_snapshot(f"geometry_api({class_name})")

        # Resolve parameters
        resolved_params = resolve_value(init_params)

        # Create the object
        if class_name == "Vector":
            if "self" in resolved_params:
                # Method call on existing vector
                obj = resolved_params.pop("self")
            elif "tuple" in resolved_params:
                obj = Vector(*resolved_params["tuple"])
            else:
                x = resolved_params.get("x", 0)
                y = resolved_params.get("y", 0)
                z = resolved_params.get("z", 0)
                obj = Vector(x, y, z)

            values = {"x": obj.x, "y": obj.y, "z": obj.z, "length": obj.Length}

        elif class_name == "Plane":
            if "name" in resolved_params:
                plane_name = resolved_params["name"]
                # Use Plane.named() classmethod for standard planes
                obj = Plane.named(plane_name)
            else:
                origin = resolved_params.get("origin", (0, 0, 0))
                xDir = resolved_params.get("xDir", (1, 0, 0))
                normal = resolved_params.get("normal", (0, 0, 1))
                if isinstance(origin, list):
                    origin = tuple(origin)
                if isinstance(xDir, list):
                    xDir = tuple(xDir)
                if isinstance(normal, list):
                    normal = tuple(normal)
                obj = Plane(origin=origin, xDir=xDir, normal=normal)

            values = {
                "origin": [obj.origin.x, obj.origin.y, obj.origin.z],
                "xDir": [obj.xDir.x, obj.xDir.y, obj.xDir.z],
                "normal": [obj.zDir.x, obj.zDir.y, obj.zDir.z]
            }

        elif class_name == "Location":
            if "plane" in resolved_params:
                plane = resolved_params["plane"]
                obj = Location(plane)
            elif "vector" in resolved_params:
                vec = resolved_params["vector"]
                obj = Location(vec)
            else:
                x = resolved_params.get("x", 0)
                y = resolved_params.get("y", 0)
                z = resolved_params.get("z", 0)
                obj = Location(Vector(x, y, z))

            try:
                t = obj.toTuple()
                values = {"translation": list(t[0]), "rotation": list(t[1])}
            except Exception:
                values = {}  # Failed to convert Location to tuple

        elif class_name == "Matrix":
            values_list = resolved_params.get("values", None)
            if values_list:
                obj = Matrix(values_list)
            else:
                obj = Matrix()
            values = {}

        elif class_name == "BoundBox":
            if "shape" in resolved_params:
                shape = resolved_params["shape"]
                if hasattr(shape, 'val'):
                    shape = shape.val()
                obj = shape.BoundingBox()
                values = {
                    "xmin": obj.xmin, "xmax": obj.xmax,
                    "ymin": obj.ymin, "ymax": obj.ymax,
                    "zmin": obj.zmin, "zmax": obj.zmax,
                    "center": [obj.center.x, obj.center.y, obj.center.z]
                }
            else:
                return error_response("BoundBox requires a shape parameter")

        else:
            return error_response(f"Unknown class: {class_name}")

        # Apply method if specified
        if method and hasattr(obj, method):
            method_func = getattr(obj, method)
            if method_params:
                resolved_method_params = resolve_value(method_params)
                obj = method_func(**resolved_method_params)
            else:
                obj = method_func()

            # Update values for the new object
            if isinstance(obj, Vector):
                values = {"x": obj.x, "y": obj.y, "z": obj.z, "length": obj.Length}

        # Generate name if not provided
        if store_as is None:
            store_as = StateManager.auto_name(class_name.lower())

        # Store the result
        StateManager.store(
            name=store_as,
            obj=obj,
            obj_type=class_name,
            operation=f"geometry_api({class_name})",
            parameters={}
        )

        show_safe(obj)

        return json.dumps({
            "status": "success",
            "name": store_as,
            "obj_type": class_name,
            "values": values
        })

    except Exception as e:
        return error_response(str(e), traceback.format_exc())


@mcp.tool(name="cq_selector_api")
async def cq_selector_api(
    operation: Literal["create", "filter", "string_syntax_help"],
    selector_class: Optional[Literal["Selector", "BinarySelector", "NearestToPointSelector", "BoxSelector", "BaseDirSelector", "ParallelDirSelector", "DirectionSelector", "PerpendicularDirSelector", "TypeSelector", "RadiusNthSelector", "CenterNthSelector", "DirectionMinMaxSelector", "DirectionNthSelector", "LengthNthSelector", "AreaNthSelector", "AndSelector", "SumSelector", "SubtractSelector", "InverseSelector", "StringSyntaxSelector"]] = None,
    selector_params: Optional[dict] = None,
    object_name: Optional[str] = None,
    shape_type: Optional[Literal["faces", "edges", "vertices", "wires", "solids"]] = None,
    selector: Optional[dict] = None,
    store_as: Optional[str] = None
) -> str:
    """
    Create selector objects for filtering shapes, or filter shapes using selectors.
    Selectors filter lists of shapes based on geometric criteria.

    IMPORTANT: Before using this tool, you MUST first call the cq_load_docs tool to load the relevant documentation for this API tool !!! 

    Args:
        operation: The operation to perform
            - "create": Create a new selector object
            - "filter": Filter shapes from an object using a selector
            - "string_syntax_help": Get help on string selector syntax
        selector_class: Selector class name (required for "create" operation)
        selector_params: Parameters for creating the selector
        object_name: Name of stored object to filter (for "filter" operation)
        shape_type: Type of shapes to filter ("faces", "edges", "vertices", "wires", "solids")
        selector: Selector to use for filtering
        store_as: Name to store the selector or filtered result

    All available selector classes: Selector, NearestToPointSelector, BoxSelector, BaseDirSelector, ParallelDirSelector, DirectionSelector, PerpendicularDirSelector, TypeSelector, RadiusNthSelector, CenterNthSelector, DirectionMinMaxSelector, DirectionNthSelector, LengthNthSelector, AreaNthSelector, BinarySelector, AndSelector, SumSelector, SubtractSelector, InverseSelector, StringSyntaxSelector

    Examples:
        Create a selector:
        {
            "operation": "create",
            "selector_class": "DirectionMinMaxSelector",
            "selector_params": {"vector": [0, 0, 1], "directionMax": true},
            "store_as": "top_face_selector"
        }

        Filter using stored selector:
        {
            "operation": "filter",
            "object_name": "my_box",
            "shape_type": "faces",
            "selector": {"_ref": "top_face_selector"},
            "store_as": "top_faces"
        }

        Filter using string selector:
        {
            "operation": "filter",
            "object_name": "my_box",
            "shape_type": "faces",
            "selector": {"string": ">Z"},
            "store_as": "top_faces"
        }

    Returns:
        JSON string with: status, name, obj_type, count (number of shapes selected)

    Error Handling:
        Returns JSON with status="error" for:
        - Unknown operation type
        - selector_class not provided for "create" operation
        - Unknown selector class name
        - object_name/shape_type/selector not provided for "filter"
        - Referenced object not found
        - Shape type not supported by object
        - Invalid selector specification
    """
    try:
        if operation == "string_syntax_help":
            help_text = """
String Selector Syntax:
  | parallel to (e.g., "|Z" for edges parallel to Z axis)
  # perpendicular to (e.g., "#Z" for faces perpendicular to Z)
  + positive direction (e.g., "+Z" for faces with normal in +Z)
  - negative direction (e.g., "-Z" for faces with normal in -Z)
  > maximize (e.g., ">Z" for topmost face)
  < minimize (e.g., "<Z" for bottommost face)
  % type selector (e.g., "%CYLINDER" for cylindrical faces)

Axes: X, Y, Z, XY, YZ, XZ or (x,y,z) for arbitrary direction

Index: [n] for nth item, [-n] for nth from end
  e.g., ">Z[-2]" for second from top face

Logical: and, or, not, exc(ept)
  e.g., "|Z and >Y" for edges parallel to Z and maximizing Y

Type selectors: PLANE, CYLINDER, CONE, SPHERE, TORUS, BEZIER, BSPLINE, REVOLUTION, EXTRUSION, OFFSET, LINE, CIRCLE, ELLIPSE, HYPERBOLA, PARABOLA
"""
            return json.dumps({
                "status": "success",
                "help": help_text
            })

        elif operation == "create":
            # Save state for undo
            StateManager.save_snapshot(f"selector_api(create {selector_class})")

            if selector_class is None:
                return error_response("selector_class is required for create operation")

            # Resolve parameters
            resolved_params = resolve_value(selector_params or {})

            # Create selector
            selector_map = {
                "Selector": Selector,
                "BinarySelector": BinarySelector,
                "NearestToPointSelector": NearestToPointSelector,
                "BoxSelector": BoxSelector,
                "BaseDirSelector": BaseDirSelector,
                "ParallelDirSelector": ParallelDirSelector,
                "DirectionSelector": DirectionSelector,
                "PerpendicularDirSelector": PerpendicularDirSelector,
                "TypeSelector": TypeSelector,
                "RadiusNthSelector": RadiusNthSelector,
                "CenterNthSelector": CenterNthSelector,
                "DirectionMinMaxSelector": DirectionMinMaxSelector,
                "DirectionNthSelector": DirectionNthSelector,
                "LengthNthSelector": LengthNthSelector,
                "AreaNthSelector": AreaNthSelector,
                "AndSelector": AndSelector,
                "SumSelector": SumSelector,
                "SubtractSelector": SubtractSelector,
                "InverseSelector": InverseSelector,
                "StringSyntaxSelector": StringSyntaxSelector
            }

            cls = selector_map.get(selector_class)
            if cls is None:
                return error_response(f"Unknown selector class: {selector_class}")

            # Handle special parameters
            if selector_class in ["NearestToPointSelector"]:
                pnt = resolved_params.get("pnt", [0, 0, 0])
                if isinstance(pnt, list):
                    pnt = Vector(*pnt)
                selector_obj = cls(pnt)
            elif selector_class in ["BoxSelector"]:
                point0 = resolved_params.get("point0", [0, 0, 0])
                point1 = resolved_params.get("point1", [1, 1, 1])
                boundingbox = resolved_params.get("boundingbox", False)
                if isinstance(point0, list):
                    point0 = Vector(*point0)
                if isinstance(point1, list):
                    point1 = Vector(*point1)
                selector_obj = cls(point0, point1, boundingbox)
            elif selector_class in ["ParallelDirSelector", "DirectionSelector", "PerpendicularDirSelector"]:
                vector = resolved_params.get("vector", [0, 0, 1])
                tolerance = resolved_params.get("tolerance", 0.0001)
                if isinstance(vector, list):
                    vector = Vector(*vector)
                selector_obj = cls(vector, tolerance)
            elif selector_class == "TypeSelector":
                typeString = resolved_params.get("typeString", "PLANE")
                selector_obj = cls(typeString)
            elif selector_class in ["RadiusNthSelector", "LengthNthSelector", "AreaNthSelector"]:
                n = resolved_params.get("n", 0)
                directionMax = resolved_params.get("directionMax", True)
                tolerance = resolved_params.get("tolerance", 0.0001)
                selector_obj = cls(n, directionMax, tolerance)
            elif selector_class == "CenterNthSelector":
                vector = resolved_params.get("vector", [0, 0, 1])
                n = resolved_params.get("n", 0)
                if isinstance(vector, list):
                    vector = Vector(*vector)
                selector_obj = cls(vector, n)
            elif selector_class == "DirectionMinMaxSelector":
                vector = resolved_params.get("vector", [0, 0, 1])
                directionMax = resolved_params.get("directionMax", True)
                tolerance = resolved_params.get("tolerance", 0.0001)
                if isinstance(vector, list):
                    vector = Vector(*vector)
                selector_obj = cls(vector, directionMax, tolerance)
            elif selector_class == "DirectionNthSelector":
                vector = resolved_params.get("vector", [0, 0, 1])
                n = resolved_params.get("n", 0)
                if isinstance(vector, list):
                    vector = Vector(*vector)
                selector_obj = cls(vector, n)
            elif selector_class in ["AndSelector", "SumSelector", "SubtractSelector"]:
                left = resolved_params.get("left")
                right = resolved_params.get("right")
                selector_obj = cls(left, right)
            elif selector_class == "InverseSelector":
                inner_selector = resolved_params.get("selector")
                selector_obj = cls(inner_selector)
            elif selector_class == "StringSyntaxSelector":
                selectorString = resolved_params.get("selectorString", ">Z")
                selector_obj = cls(selectorString)
            else:
                selector_obj = cls(**resolved_params)

            # Generate name if not provided
            if store_as is None:
                store_as = StateManager.auto_name("selector")

            # Store the selector
            StateManager.store(
                name=store_as,
                obj=selector_obj,
                obj_type="Selector",
                operation=f"selector_api(create {selector_class})",
                parameters={"selector_class": selector_class}
            )

            return json.dumps({
                "status": "success",
                "name": store_as,
                "obj_type": "Selector",
                "selector_class": selector_class
            })

        elif operation == "filter":
            # Save state for undo
            StateManager.save_snapshot(f"selector_api(filter)")

            if object_name is None:
                return error_response("object_name is required for filter operation")
            if shape_type is None:
                return error_response("shape_type is required for filter operation")

            # Get the object to filter
            obj = StateManager.get(object_name)

            # Handle selector
            if selector is None:
                return error_response("selector is required for filter operation")

            # Check if it's a string selector
            if isinstance(selector, dict) and "string" in selector:
                selector_obj = selector["string"]  # This is a string
            else:
                resolved_selector = resolve_value(selector)
                if isinstance(resolved_selector, Selector):
                    # Use the selector directly
                    selector_obj = resolved_selector  # This is a Selector
                else:
                    return error_response("Invalid selector")

            # Apply selection
            if isinstance(obj, Workplane):
                method = getattr(obj, shape_type)
                result = method(selector_obj)
            else:
                # For direct shapes, we need to get the shapes and filter
                method_name = shape_type.capitalize()
                if hasattr(obj, method_name):
                    shapes = getattr(obj, method_name)()
                    if isinstance(selector_obj, str):
                        # Apply string selector
                        wp = Workplane().add(obj)
                        method = getattr(wp, shape_type)
                        result = method(selector_obj)
                    else:
                        # Apply selector object
                        if hasattr(shapes, '__iter__'):
                            result = selector_obj.filter(list(shapes))
                        else:
                            result = shapes
                else:
                    return error_response(f"Cannot get {shape_type} from {type(obj).__name__}")

            # Generate name if not provided
            if store_as is None:
                store_as = StateManager.auto_name(f"{shape_type}_selection")

            # Determine object type and count
            obj_type = determine_obj_type(result)

            # Count selected items
            count = 0
            if isinstance(result, Workplane):
                try:
                    count = len(result.vals())
                except Exception:
                    count = 0  # Failed to get count from Workplane
            elif hasattr(result, '__len__'):
                count = len(result)
            elif result is not None:
                count = 1  # Single non-iterable result

            # Store the result
            StateManager.store(
                name=store_as,
                obj=result,
                obj_type=obj_type,
                parent=object_name,
                operation=f"selector_api(filter {shape_type})",
                parameters={"shape_type": shape_type}
            )

            show_safe(result)

            return json.dumps({
                "status": "success",
                "name": store_as,
                "obj_type": obj_type,
                "count": count
            })

        else:
            return error_response(f"Unknown operation: {operation}")

    except Exception as e:
        return error_response(str(e), traceback.format_exc())


@mcp.tool(name="cq_free_function_api")
async def cq_free_function_api(
    function: Literal["edgeOn", "wireOn", "wire", "face", "shell", "solid", "compound", "vertex", "segment", "polyline", "polygon", "rect", "spline", "circle", "ellipse", "plane", "box", "cylinder", "sphere", "torus", "cone", "text", "fuse", "cut", "intersect", "imprint", "split", "fill", "clean", "cap", "fillet", "chamfer", "extrude", "revolve", "offset", "sweep", "loft", "check", "closest", "setThreads", "project", "faceOn", "isSubshape"],
    params: dict,
    store_as: Optional[str] = None
) -> str:
    """
    Create shapes using free functions - no hidden state, explicit operations.
    The free function API provides stateless shape construction where all operations
    are explicit functions rather than chained methods. It can be used with Direct API's and Geometry API's classes as well.

    IMPORTANT: Before using this tool, you MUST first call the cq_load_docs tool to load the relevant documentation for this API tool !!! 

    Args:
        function: Name of the free function to call
            Primitives (1D): circle, ellipse, segment, spline, polyline
            Primitives (2D): rect, polygon, plane
            Primitives (3D): box, sphere, cylinder, cone, torus, vertex
            Construction: wire, face, shell, solid, compound, fill, cap
            Operations: extrude, loft, revolve, sweep
            Boolean: fuse, cut, intersect, split
            Modification: fillet, chamfer, offset
            Text: text
            Parametric: edgeOn, wireOn, faceOn
            All available functions: edgeOn, wireOn, wire, face,  shell, solid, compound, vertex, segment, polyline, polygon, rect, spline, circle, ellipse, plane, box, cylinder, sphere, torus, cone, text, fuse, cut, intersect, imprint, split, fill, clean, cap, fillet, chamfer, extrude, revolve, offset, sweep, loft, check, closest, setThreads, project, faceOn, isSubshape
        params: Function parameters. Can include references and type constructions.
        store_as: Name to store the result. Auto-generated if not provided.

    Returns:
        JSON string with: status, name, obj_type, properties

    Error Handling:
        Returns JSON with status="error" for:
        - Free function API not available (wrong CadQuery version)
        - Unknown function name
        - Invalid function parameters
        - CadQuery operation failures
    """
    try:
        if not FREE_FUNCTIONS_AVAILABLE:
            return json.dumps({
                "status": "error",
                "error": "Free function API not available in this CadQuery version. Use cq_fluent_api or cq_direct_api instead."
            })

        # Save state for undo
        StateManager.save_snapshot(f"free_function_api({function})")

        # Resolve parameters
        resolved_params = resolve_value(params)

        # Map function names to actual functions
        function_map = {
            "edgeOn": edgeOn,
            "wireOn": wireOn,
            "imprint": imprint,
            "clean": clean,
            "cap": cap,
            "check": check,
            "closest": closest,
            "setThreads": setThreads,
            "project": project,
            "faceOn": faceOn,
            "isSubshape": isSubshape,
            "circle": circle,
            "ellipse": ellipse,
            "rect": rect,
            "polygon": polygon,
            "polyline": polyline,
            "segment": segment,
            "spline": spline,
            "vertex": vertex,
            "box": box,
            "sphere": sphere,
            "cylinder": cylinder,
            "cone": cone,
            "torus": torus,
            "plane": plane,
            "wire": wire,
            "face": face,
            "shell": shell,
            "solid": solid,
            "compound": compound,
            "fill": fill,
            "extrude": extrude,
            "loft": loft,
            "revolve": revolve,
            "sweep": sweep,
            "fuse": fuse,
            "cut": cut,
            "intersect": intersect,
            "split": split,
            "fillet": fillet,
            "chamfer": chamfer,
            "offset": offset,
            "text": text
        }

        func = function_map.get(function)
        if func is None:
            return error_response(f"Unknown function: {function}")

        # Call the function
        result = func(**resolved_params)

        # Generate name if not provided
        if store_as is None:
            store_as = StateManager.auto_name(function)

        # Determine object type
        obj_type = determine_obj_type(result)

        # Store the result
        StateManager.store(
            name=store_as,
            obj=result,
            obj_type=obj_type,
            operation=f"free_function_api({function})",
            parameters={"function": function}
        )

        show_safe(result)

        # Get properties
        props = get_object_properties(result)

        return json.dumps({
            "status": "success",
            "name": store_as,
            "obj_type": obj_type,
            "properties": props
        })

    except Exception as e:
        return error_response(str(e), traceback.format_exc())


@mcp.tool(name="cq_state_manager")
async def cq_state_manager(
    operation: Literal["list", "get_info", "set_current", "delete", "clear", "hierarchy", "history", "undo", "redo", "undo_status"],
    operation_params: Optional[dict] = None
) -> str:
    """
    Manage the CAD modeling session state.
    Use this to list available objects, inspect object details, delete objects,
    view the object hierarchy, access operation history, and perform undo/redo.

    Args:
        operation: The state management operation to perform
            - "list": List all stored objects
            - "get_info": Get detailed information about an object
            - "set_current": Set the active/current object
            - "delete": Remove an object from storage
            - "clear": Clear all stored objects
            - "hierarchy": View parent-child relationships
            - "history": View operation history
            - "undo": Undo the last operation # IMPORTANT: This method currently does not work so don't use it!
            - "redo": Redo a previously undone operation # IMPORTANT: This method currently does not work so don't use it!
            - "undo_status": Check undo/redo availability # IMPORTANT: This method currently does not work so don't use it!
        operation_params: Operation-specific parameters
            For "list": obj_type (optional) - filter by object type
            For "get_info": name (required) - object name
            For "set_current": name (required) - object name
            For "delete": name (required), cascade (optional)
            For "clear": confirm (required) - must be true
            For "history": limit (optional) - max entries

    Returns:
        JSON string with operation-specific results

    Error Handling:
        Returns JSON with status="error" for:
        - Unknown operation type
        - Missing required operation_params (name, confirm, etc.)
        - Referenced object not found
        - Nothing to undo/redo when attempting those operations
    """
    try:
        if operation_params is None:
            operation_params = {}

        if operation == "list":
            obj_type = operation_params.get("obj_type", None)
            objects = StateManager.list_objects(obj_type)
            return json.dumps({
                "status": "success",
                "count": len(objects),
                "objects": objects,
                "current": _session_state["current"]
            })

        elif operation == "get_info":
            name = operation_params.get("name")
            if name is None:
                return error_response("name is required")

            metadata = StateManager.get_metadata(name)
            obj = StateManager.get(name)
            props = get_object_properties(obj)

            return json.dumps({
                "status": "success",
                "info": metadata.to_dict(),
                "properties": props
            })

        elif operation == "set_current":
            name = operation_params.get("name")
            if name is None:
                return error_response("name is required")

            StateManager.set_current(name)

            # Visualize the newly set current object
            obj = StateManager.get(name)
            show_safe(obj)

            return json.dumps({
                "status": "success",
                "current": name
            })

        elif operation == "delete":
            name = operation_params.get("name")
            if name is None:
                return error_response("name is required")

            cascade = operation_params.get("cascade", False)
            StateManager.save_snapshot(f"delete({name})")
            StateManager.delete(name, cascade)

            # Visualize current state after deletion
            current_name = _session_state["current"]
            if current_name and current_name in _session_state["objects"]:
                obj = StateManager.get()  # Gets current by default
                show_safe(obj)
            else:
                # Current is None, show empty workspace
                empty = Workplane("XY")
                show_safe(empty)

            return json.dumps({
                "status": "success",
                "deleted": name,
                "cascade": cascade
            })

        elif operation == "clear":
            confirm = operation_params.get("confirm", False)
            if not confirm:
                return error_response("Set confirm=true to clear all objects")

            StateManager.save_snapshot("clear")
            StateManager.clear()

            # Show empty workspace since all objects cleared
            empty = Workplane("XY")
            show_safe(empty)

            return json.dumps({
                "status": "success",
                "message": "All objects cleared"
            })

        elif operation == "hierarchy":
            hierarchy = StateManager.get_hierarchy()
            return json.dumps({
                "status": "success",
                "hierarchy": hierarchy
            })

        elif operation == "history":
            limit = operation_params.get("limit", 50)
            history = StateManager.get_history(limit)
            return json.dumps({
                "status": "success",
                "count": len(history),
                "history": history
            })

        elif operation == "undo":
            description = StateManager.undo()
            if description:
                # Visualize restored current state
                current_name = _session_state["current"]
                empty = Workplane("XY")
                show_safe(empty)
                # if current_name and current_name in _session_state["objects"]:
                #     obj = StateManager.get()  # Gets current by default
                #     show_safe(obj)
                # else:
                #     # Restored to empty state
                #     empty = Workplane("XY")
                #     show_safe(empty)

                return json.dumps({
                    "status": "success",
                    "undone": description
                })
            else:
                return error_response("Nothing to undo")

        elif operation == "redo":
            description = StateManager.redo()
            if description:
                # Visualize restored current state
                current_name = _session_state["current"]
                if current_name and current_name in _session_state["objects"]:
                    obj = StateManager.get()  # Gets current by default
                    show_safe(obj)
                else:
                    # Restored to empty state
                    empty = Workplane("XY")
                    show_safe(empty)

                return json.dumps({
                    "status": "success",
                    "redone": description
                })
            else:
                return error_response("Nothing to redo")

        elif operation == "undo_status":
            return json.dumps({
                "status": "success",
                "can_undo": StateManager.can_undo(),
                "can_redo": StateManager.can_redo(),
                "undo_count": len(_session_state["undo_stack"]),
                "redo_count": len(_session_state["redo_stack"])
            })

        else:
            return error_response(f"Unknown operation: {operation}")

    except Exception as e:
        return error_response(str(e), traceback.format_exc())


@mcp.tool(name="cq_import_export")
async def cq_import_export(
    operation: Literal["import", "export"],
    file_path: str,
    object_name: Optional[str] = None,
    store_as: Optional[str] = None,
    export_options: Optional[dict] = None
) -> str:
    """
    Import STEP files and export CadQuery objects to STEP format.
    Supports both Workplane/Shape objects and Assembly objects with full assembly export options.
    Includes a comprehensive catalog of pre-made parts for rapid prototyping.

    AVAILABLE PARTS CATALOG:

    Boards:
      Arduino:
        - Arduino Nano: C:\\Users\\user\\Desktop\\Auto\\Playground\\cadquery\\cq2\\Parts\\Boards\\arduino_nano_r3.step
        - Arduino UNO: C:\\Users\\user\\Desktop\\Auto\\Playground\\cadquery\\cq2\\Parts\\Boards\\arduino_uno.step
        - Arduino Mega: C:\\Users\\user\\Desktop\\Auto\\Playground\\cadquery\\cq2\\Parts\\Boards\\arduino_mega.step
        - Arduino Nano Extension Board: C:\\Users\\user\\Desktop\\Auto\\Playground\\cadquery\\cq2\\Parts\\Boards\\arduino_nano_expansion_shield.step
        - Arduino Display 2.4 TFT SPI 240x320: C:\\Users\\user\\Desktop\\Auto\\Playground\\cadquery\\cq2\\Parts\\Displays\\arduino_2-4_tft_spi_240x320.step

      Raspberry Pi:
        - Raspberry PI Pico: C:\\Users\\user\\Desktop\\Auto\\Playground\\cadquery\\cq2\\Parts\\Boards\\raspberry_pi_pico_r3.step
        - Raspberry PI 4B: C:\\Users\\user\\Desktop\\Auto\\Playground\\cadquery\\cq2\\Parts\\Boards\\raspberry_pi_4B.step # IMPORTANT: This file is extremely large, so try to avoid using it. Use Raspberry PI Pico or Arduino boards instead.
        - Raspberry PI Camera (170 degree): C:\\Users\\user\\Desktop\\Auto\\Playground\\cadquery\\cq2\\Parts\\Cameras\\raspberry_pi_camera_170_angle.step
        - Raspberry PI Display 3.5 TFT SPI 480x320: C:\\Users\\user\\Desktop\\Auto\\Playground\\cadquery\\cq2\\Parts\\Displays\\raspberry_pi_3.5_tft_display_spi_480x320.step

      LCDs:
        - LCD 4x20 backlight: C:\\Users\\user\\Desktop\\Auto\\Playground\\cadquery\\cq2\\Parts\\Displays\\generic_lcd_4x20_backlight.step
        - Smart LCD Ramps: C:\\Users\\user\\Desktop\\Auto\\Playground\\cadquery\\cq2\\Parts\\Displays\\smart_lcd_ramps.step

    Batteries and Cells:
      18650 Lithium-Ion:
        - Single cell (3.7-4V): C:\\Users\\user\\Desktop\\Auto\\Playground\\cadquery\\cq2\\Parts\\Batteries\\18650_single_cell.step
        - Single cell with holder (3.7-4V): C:\\Users\\user\\Desktop\\Auto\\Playground\\cadquery\\cq2\\Parts\\Batteries\\18650_single_cell_holder_with_cell.step
        - 3x cell holder without cells (11-12V): C:\\Users\\user\\Desktop\\Auto\\Playground\\cadquery\\cq2\\Parts\\Batteries\\18650_holder_3_cells_12v.step
        - 3x cell holder with cells (11-12V): C:\\Users\\user\\Desktop\\Auto\\Playground\\cadquery\\cq2\\Parts\\Batteries\\18650_holder_3_cells_12v_with_cells.step
        - 4x cell holder without cells (14-16V): C:\\Users\\user\\Desktop\\Auto\\Playground\\cadquery\\cq2\\Parts\\Batteries\\18650_holder_4_cells_16v.step
        - 4x cell holder with cells (14-16V): C:\\Users\\user\\Desktop\\Auto\\Playground\\cadquery\\cq2\\Parts\\Batteries\\18650_holder_4_cells_16v_with_cells.step

      AAA Cells:
        - Single cell (1.5V): C:\\Users\\user\\Desktop\\Auto\\Playground\\cadquery\\cq2\\Parts\\Batteries\\battery-AAA.step
        - 4x cell holder without cells (6V): C:\\Users\\user\\Desktop\\Auto\\Playground\\cadquery\\cq2\\Parts\\Batteries\\battery-holder-4-AAA.step

      9V Battery:
        - 9V battery: C:\\Users\\user\\Desktop\\Auto\\Playground\\cadquery\\cq2\\Parts\\Batteries\\battery_9v.step

      LiPo:
        - 3.7V 1200mAh: C:\\Users\\user\\Desktop\\Auto\\Playground\\cadquery\\cq2\\Parts\\Batteries\\lipo_3_7v_1200mAh.step

    Motors:
      DC Motors:
        - L-shaped yellow gear motor: C:\\Users\\user\\Desktop\\Auto\\Playground\\cadquery\\cq2\\Parts\\Motors\\Yellow_gearmotor_L.step
        - Straight yellow gear motor: C:\\Users\\user\\Desktop\\Auto\\Playground\\cadquery\\cq2\\Parts\\Motors\\Yellow_gearmotor_straight.step
        - Pololu 298 1.6V: C:\\Users\\user\\Desktop\\Auto\\Playground\\cadquery\\cq2\\Parts\\Motors\\Pololu_298_1_6V.step
        - Metal gear motor 37mm: C:\\Users\\user\\Desktop\\Auto\\Playground\\cadquery\\cq2\\Parts\\Motors\\Metal_gearmotor_37mm.step

      Stepper:
        - NEMA 17: C:\\Users\\user\\Desktop\\Auto\\Playground\\cadquery\\cq2\\Parts\\Motors\\NEMA_17_with_connector.step
        - NEMA 23: C:\\Users\\user\\Desktop\\Auto\\Playground\\cadquery\\cq2\\Parts\\Motors\\NEMA_23_with_connector.step

      Servo:
        - EMAX ES08A: C:\\Users\\user\\Desktop\\Auto\\Playground\\cadquery\\cq2\\Parts\\Motors\\Servo_emax_es08A.step

    Wheels:
      - 65mm yellow wheel (for yellow gear motors): C:\\Users\\user\\Desktop\\Auto\\Playground\\cadquery\\cq2\\Parts\\Motors\\Yellow_wheel_65mm.step

    Sensors:
      IR Sensor:
        - IR sensor: C:\\Users\\user\\Desktop\\Auto\\Playground\\cadquery\\cq2\\Parts\\Sensors\\IR_sensor.step

      Ultrasonic Sensor:
        - HC-SR04: C:\\Users\\user\\Desktop\\Auto\\Playground\\cadquery\\cq2\\Parts\\Sensors\\Ultrasonic_sensor_HC_SR04.step

    Other:
      Robotic Arm Grippers:
        - Gripper with 2 claws: C:\\Users\\user\\Desktop\\Auto\\Playground\\cadquery\\cq2\\Parts\\Other\\robotic_arm_gripper_1.step
        - Gripper with 4 claws: C:\\Users\\user\\Desktop\\Auto\\Playground\\cadquery\\cq2\\Parts\\Other\\robotic_arm_gripper_2.stp

    Args:
        operation: Operation to perform ("import" or "export")
        file_path: Path to STEP file
            - Import: Full path to source STEP file to import
            - Export: Filename or full path. If just a filename (e.g., "box.step"),
              automatically saves to C:\\Users\\user\\Desktop\\mcp-dev\\cadquery\\exports\\
              If full path provided, uses that path as-is.
        object_name: Name of object in StateManager to export (required for export operation)
        store_as: Custom name for imported object (optional for import, auto-generated if not provided)
        export_options: Export configuration (optional for export, used with Assembly objects):
            - mode: "default" (multi-part) or "fused" (single combined shape)
            - glue: True/False - Apply gluing when mode="fused"
            - write_pcurves: True/False - Include parametric curve data (default True)

    Returns:
        JSON string with status and operation details

        Import success response:
        {
            "status": "success",
            "operation": "import",
            "name": "imported_1",
            "obj_type": "Workplane",
            "file_path": "/path/to/file.step",
            "properties": {"volume": 1000, "area": 600, "center": [0,0,0], "bounding_box": {...}}
        }

        Export success response:
        {
            "status": "success",
            "operation": "export",
            "object_name": "my_part",
            "file_path": "/path/to/output.step",
            "export_options": {"mode": "default", "glue": False, "write_pcurves": True}
        }

    Examples:
        Import Arduino Nano:
        {
            "operation": "import",
            "file_path": "C:\\\\Users\\\\user\\\\Desktop\\\\Auto\\\\Playground\\\\cadquery\\\\cq2\\\\Parts\\\\Boards\\\\arduino_nano_r3.step",
            "store_as": "arduino_nano"
        }

        Import with auto-generated name:
        {
            "operation": "import",
            "file_path": "C:\\\\Users\\\\user\\\\Desktop\\\\Auto\\\\Playground\\\\cadquery\\\\cq2\\\\Parts\\\\Motors\\\\NEMA_17_with_connector.step"
        }

        Export with just filename (saves to exports directory):
        {
            "operation": "export",
            "object_name": "my_box",
            "file_path": "box.step"
        }

        Export to specific path:
        {
            "operation": "export",
            "object_name": "my_assembly",
            "file_path": "C:\\\\exports\\\\custom\\\\assembly.step"
        }

        Export Assembly with fused mode:
        {
            "operation": "export",
            "object_name": "robot_assembly",
            "file_path": "robot.step",
            "export_options": {
                "mode": "fused",
                "glue": True,
                "write_pcurves": False
            }
        }

    Error Handling:
        Returns JSON with status="error" for:
        - Unknown operation type (not "import" or "export")
        - File not found during import
        - Invalid file path or permissions during export
        - Missing object_name for export operation
        - Object not found in StateManager
        - File I/O errors
        - Invalid STEP file format
    """
    try:
        # Save state for undo
        StateManager.save_snapshot(f"import_export({operation})")

        if operation == "import":
            # ===== IMPORT OPERATION =====

            # 1. Validate file exists
            if not os.path.exists(file_path):
                return error_response(f"File not found: {file_path}")

            # 2. Import the STEP file
            try:
                imported_obj = cq.importers.importStep(file_path)
                show_safe(imported_obj)
            except Exception as import_error:
                return error_response(f"Failed to import STEP file: {str(import_error)}")

            # 3. Generate storage name if not provided
            if store_as is None:
                store_as = StateManager.auto_name("imported")

            # 4. Determine object type
            obj_type = determine_obj_type(imported_obj)

            # 5. Store in StateManager
            StateManager.store(
                name=store_as,
                obj=imported_obj,
                obj_type=obj_type,
                operation="import_export(import)",
                parameters={"file_path": file_path}
            )

            # 6. Visualize in OCP viewer
            show_safe(imported_obj)

            # 7. Get properties
            props = get_object_properties(imported_obj)

            # 8. Return success response
            return json.dumps({
                "status": "success",
                "operation": "import",
                "name": store_as,
                "obj_type": obj_type,
                "file_path": file_path,
                "properties": props
            })

        elif operation == "export":
            # ===== EXPORT OPERATION =====

            # 1. Validate object_name provided
            if object_name is None:
                return error_response("object_name is required for export operation")

            # 2. Get object from StateManager
            try:
                obj = StateManager.get(object_name)
            except ValueError as e:
                return error_response(f"Object not found: {str(e)}")

            # 3. Process file_path - auto-prepend exports directory if just filename
            EXPORTS_DIR = r"C:\Users\user\Desktop\mcp-dev\cadquery\exports"

            # Check if file_path is just a filename (no directory separators)
            if not os.path.dirname(file_path):
                # Just a filename, prepend exports directory
                file_path = os.path.join(EXPORTS_DIR, file_path)

            # 4. Ensure the export directory exists
            export_dir = os.path.dirname(file_path)
            if export_dir and not os.path.exists(export_dir):
                try:
                    os.makedirs(export_dir, exist_ok=True)
                except Exception as dir_error:
                    return error_response(f"Failed to create export directory: {str(dir_error)}")

            # 5. Determine object type
            obj_type = determine_obj_type(obj)

            # 6. Perform export based on object type
            if isinstance(obj, Assembly):
                # Assembly export with options
                resolved_options = export_options or {}

                # Extract Assembly-specific options
                mode = resolved_options.get("mode", "default")
                glue = resolved_options.get("glue", False)
                write_pcurves = resolved_options.get("write_pcurves", True)

                # Build opt dict for Assembly.save()
                opt = {}
                if not write_pcurves:
                    opt["write_pcurves"] = False

                try:
                    if mode == "fused":
                        # Export as fused assembly
                        obj.save(file_path, mode="fused", glue=glue, **opt)
                    else:
                        # Export as default assembly (separate parts)
                        obj.save(file_path, **opt)
                except Exception as export_error:
                    return error_response(f"Assembly export failed: {str(export_error)}")

            else:
                # Workplane or Shape export
                try:
                    if hasattr(obj, 'export'):
                        # Workplane has .export() method
                        obj.export(file_path)
                    else:
                        # Use cq.exporters for direct shapes
                        cq.exporters.export(obj, file_path, exportType="STEP")
                except Exception as export_error:
                    return error_response(f"Export failed: {str(export_error)}")

            # 7. Return success response
            return json.dumps({
                "status": "success",
                "operation": "export",
                "object_name": object_name,
                "obj_type": obj_type,
                "file_path": file_path,
                "export_options": export_options or {}
            })

        else:
            return error_response(f"Unknown operation: {operation}")

    except Exception as e:
        return error_response(str(e), traceback.format_exc())


@mcp.tool(name="cq_load_docs")
async def cq_load_docs(
    api_category: Literal["fluent-api", "direct-api", "geometry-api",
                          "selectors-api", "free-functions", "overview"],
    class_names: Optional[List[str]] = None,
    load_all: bool = False
) -> str:
    """
    Load MCP documentation resources into the conversation context.
    Use this tool to explore available documentation or load multiple docs at once.

    Args:
        api_category: Which API category to load docs from
            - "fluent-api": Workplane, Sketch, Assembly classes (IMPORTANT: Always load the docs of the Workplane class separately in a separate tool call)
            - "direct-api": Shape, Vertex, Edge, Wire, Face, Shell, Solid, Compound, CompSolid classes
            - "geometry-api": Vector, Plane, Location, Matrix, BoundBox classes
            - "selectors-api": All selector classes for filtering shapes
            - "free-functions": Free function API (all functions in one doc)
            - "overview": High-level overview of this MCP server (this is IMPORTANT and MUST be loaded first at the beginning of the conversation before anything else) (When requesting this set load_all = True)
        class_names: Specific class/selector names to load (e.g., ["Workplane", "Sketch"])
            If None and load_all=False, returns available classes for the category
        load_all: Load all documentation for the entire category (Default: False) (IMPORTANT: Always set this to True when requesting the 'overview' documentation)

    Returns:
        JSON string with: status, documentation (markdown text), loaded (list of URIs)

    Examples:
        Load overview docs:
            api_category="overview", load_all=True

        Load Workplane docs:
            api_category="fluent-api", class_names=["Workplane"]

        Load all geometry docs:
            api_category="geometry-api", load_all=True

        Load specific selectors:
            api_category="selectors-api", class_names=["DirectionMinMaxSelector", "TypeSelector"]

        List available classes:
            api_category="direct-api" (without class_names or load_all)
    """
    try:
        # Define available classes for each category
        category_classes = {
            "fluent-api": ["Workplane", "Sketch", "Assembly"],
            "direct-api": ["Shape", "Vertex", "Edge", "Wire", "Face", "Shell", "Solid",
                          "Compound", "CompSolid", "Mixin1D", "Mixin3D"],
            "geometry-api": ["Vector", "Plane", "Location", "Matrix", "BoundBox"],
            "selectors-api": ["Selector", "NearestToPointSelector", "BoxSelector",
                            "BaseDirSelector", "ParallelDirSelector", "DirectionSelector",
                            "PerpendicularDirSelector", "TypeSelector", "RadiusNthSelector",
                            "CenterNthSelector", "DirectionMinMaxSelector", "DirectionNthSelector",
                            "LengthNthSelector", "AreaNthSelector", "BinarySelector",
                            "AndSelector", "SumSelector", "SubtractSelector", "InverseSelector",
                            "StringSyntaxSelector"],
            "free-functions": ["all"],
            "overview": ["overview"]
        }

        # Get available classes for this category
        available = category_classes.get(api_category, [])

        # If no specific request, return available classes
        if not class_names and not load_all:
            return json.dumps({
                "status": "success",
                "api_category": api_category,
                "available_classes": available,
                "message": "Specify class_names or set load_all=True to load documentation"
            })

        # Determine which classes to load
        if load_all:
            classes_to_load = available
        else:
            classes_to_load = class_names or []

        # Build URIs
        resource_uris = []
        if api_category == "overview":
            resource_uris = ["cq://docs/overview"]
        else:
            for class_name in classes_to_load:
                if class_name in available:
                    resource_uris.append(f"cq://docs/{api_category}/{class_name}")

        # Load documentation
        if not resource_uris:
            return json.dumps({
                "status": "error",
                "error": f"No valid class names found. Available: {available}"
            })

        docs_content = load_relevant_docs_direct(resource_uris)

        return json.dumps({
            "status": "success",
            "api_category": api_category,
            "loaded": resource_uris,
            "documentation": docs_content
        })

    except Exception as e:
        return error_response(str(e), traceback.format_exc())


@mcp.tool(name="cq_feedback")
async def cq_feedback(
    object_name: str | None = None,
    views: list[str] = ["isometric_front_right", "front", "top", "right"],
    custom_projections: list[tuple[float, float, float]] | None = None,
    svg_width: int = 400,
    svg_height: int = 300,
    show_hidden_lines: bool = False,
    show_axes: bool = False,
    perspective_focus: float | None = None,
    text_detail_level: str = "comprehensive",
    include_qualitative_features: bool = True,
    include_visual_feedback: bool = True,
    include_text_feedback: bool = True,
    image_format: str = "png"
) -> list:
    """
    Get comprehensive multi-angle visual and text feedback for a CadQuery object.

    This tool provides detailed analysis of CAD models including:
    - Multi-angle visual feedback (PNG or SVG exports from 10 standard views + custom angles)
    - Detailed text analysis (topology, features, properties, or STEP export)

    Args:
        object_name: Name of object to analyze. If None, uses current object.

        views: List of standard view names to export. Available views:
            - Orthographic: "front", "back", "left", "right", "top", "bottom"
            - Isometric: "isometric_front_right", "isometric_front_left",
                        "isometric_back_right", "isometric_back_left"
            Default: ["isometric_front_right", "front", "top", "right"]

        custom_projections: List of custom (x, y, z) viewing direction tuples.
            Example: [(1, 1, 1), (0, 0, 1)] for custom isometric and top views.
            These are added to the standard views.

        svg_width: Width of exported images in pixels (default: 400)
        svg_height: Height of exported images in pixels (default: 300)

        show_hidden_lines: Render hidden edges as dashed lines (default: False)
        show_axes: Show coordinate axes in the view (default: False)

        perspective_focus: Optional focus distance for perspective projection.
            If None, uses orthographic projection. Typical value: 100.0

        text_detail_level: Level of text analysis detail:
            - "basic": Topology counts, bounding box, basic properties
            - "standard": + edge/face type analysis
            - "comprehensive": + features, vertices, full analysis (default)
            - "step_text": Export model as STEP ISO 10303-21 format # This can be really large and may cause issues with the conversation context.

        include_qualitative_features: Add human-readable feature descriptions
            (only applies when text_detail_level="comprehensive") (default: True)

        include_visual_feedback: Generate multi-angle visual exports (default: True)
        include_text_feedback: Generate text analysis (default: True)

        image_format: Export format for visual feedback:
            - "png": PNG images (requires Playwright or svglib)
            - "svg": SVG vector graphics (always available)

    Returns:
        JSON string with:
        - status: "success" or "error"
        - visual_feedback: Dict of view names to base64-encoded images
        - text_feedback: Detailed text description or STEP export
        - metadata: Format info, object type, library availability

    Examples:
        # Get default feedback (4 views as PNG + STEP export)
        cq_feedback()

        # Text-only analysis with comprehensive detail
        cq_feedback(
            include_visual_feedback=False,
            text_detail_level="comprehensive",
            include_qualitative_features=True
        )

        # Visual-only with all 10 standard views
        cq_feedback(
            views=["front", "back", "left", "right", "top", "bottom",
                   "isometric_front_right", "isometric_front_left",
                   "isometric_back_right", "isometric_back_left"],
            include_text_feedback=False
        )

        # Custom viewing angles with perspective
        cq_feedback(
            custom_projections=[(1, 1, 1), (-1, 1, 1)],
            perspective_focus=100.0,
            show_hidden_lines=True
        )

        # Analyze named object with SVG export
        cq_feedback(
            object_name="my_assembly",
            image_format="svg",
            text_detail_level="standard"
        )
    """
    try:
        # Get object using StateManager
        if object_name:
            obj = StateManager.get(object_name)
        else:
            obj = StateManager.get()  # Gets current object

        # Unwrap CQObject if needed
        if isinstance(obj, CQObject):
            cq_obj = obj.obj
        else:
            cq_obj = obj

        # Initialize result
        result = {
            "status": "success",
            "metadata": {
                "object_type": type(cq_obj).__name__,
                "image_format": image_format,
                "png_library": _PNG_LIBRARY if _PNG_CONVERSION_AVAILABLE else None
            }
        }

        # Generate visual feedback
        if include_visual_feedback:
            try:
                # Merge views and custom_projections
                all_views = views.copy() if views else []
                if custom_projections:
                    all_views.extend(custom_projections)
                if not all_views:
                    # Default to isometric if nothing specified
                    all_views = ["isometric_front_right"]

                visual_result = await _export_multiple_views(
                    cq_obj,
                    all_views,
                    width=svg_width,
                    height=svg_height,
                    show_hidden=show_hidden_lines,
                    show_axes=show_axes,
                    format=image_format,
                    focus=perspective_focus
                )
                result["visual_feedback"] = visual_result
                result["metadata"]["view_count"] = len(visual_result)

                # Add warnings if PNG requested but not available
                if image_format == "png" and not _PNG_CONVERSION_AVAILABLE:
                    result["metadata"]["warning"] = (
                        "PNG conversion libraries not available. "
                        "Install 'playwright' (recommended) or 'svglib + reportlab + pillow' for PNG support. "
                        "Falling back to SVG export."
                    )
            except Exception as ve:
                result["visual_feedback_error"] = str(ve)
                result["metadata"]["visual_feedback_status"] = "failed"

        # Generate text feedback
        if include_text_feedback:
            try:
                text_result = _generate_detailed_description(
                    cq_obj,
                    detail_level=text_detail_level,
                    enable_qualitative=include_qualitative_features
                )
                result["text_feedback"] = text_result
                result["metadata"]["text_detail_level"] = text_detail_level
            except Exception as te:
                result["text_feedback_error"] = str(te)
                result["metadata"]["text_feedback_status"] = "failed"

        # Build response as list of content items
        response_items = []

        # Add text feedback first (if enabled)
        if include_text_feedback and "text_feedback" in result:
            # Keep text feedback as-is (convert dict to JSON string)
            text_content = json.dumps(result["text_feedback"], indent=2)
            response_items.append(text_content)

        # Add images (if enabled)
        if include_visual_feedback and "visual_feedback" in result:
            for view_name, view_data in result["visual_feedback"].items():
                if "png_bytes" in view_data and not view_data.get("error"):
                    # Add caption for each view
                    caption = f"View: {view_data.get('label', view_name)}"
                    response_items.append(caption)

                    # Add the actual image
                    response_items.append(
                        MCPImage(data=view_data["png_bytes"], format="png")
                    )
                elif "svg" in view_data and not view_data.get("error"):
                    # SVG format (text/XML, not Image object)
                    response_items.append(f"SVG View {view_name}:\n{view_data['svg']}")
                elif "error" in view_data:
                    response_items.append(f"Error rendering {view_name}: {view_data['error']}")

        # Add warnings if any
        if "metadata" in result and "warning" in result["metadata"]:
            response_items.append(f"Warning: {result['metadata']['warning']}")

        # If visual feedback failed, add error message
        if "visual_feedback_error" in result:
            response_items.append(f"Visual feedback error: {result['visual_feedback_error']}")

        # If text feedback failed, add error message
        if "text_feedback_error" in result:
            response_items.append(f"Text feedback error: {result['text_feedback_error']}")

        return response_items

    except Exception as e:
        # Return error as list (matching new return type)
        error_msg = f"Error generating feedback: {str(e)}"
        if DEBUG_MODE:
            error_msg += f"\n\nTraceback:\n{traceback.format_exc()}"
        return [error_msg]


# Note: DOCS_DIR and load_doc_file are now defined earlier in the file
# (after determine_obj_type function) to avoid duplication


# ============================================================================
# FLUENT API RESOURCES
# ============================================================================
# The Fluent API provides chainable methods for building 3D models through
# Workplane, Assembly, and Sketch classes.
# ============================================================================


@mcp.resource("cq://docs/fluent-api/Workplane")
def get_workplane_docs() -> str:
    """cq Workplane class documentation - main interface for building 3D models."""
    return load_doc_file(os.path.join(DOCS_DIR, "fluent-api", "Workplane.md"))


@mcp.resource("cq://docs/fluent-api/Assembly")
def get_assembly_docs() -> str:
    """cq Assembly class documentation - managing assemblies of parts with constraints."""
    return load_doc_file(os.path.join(DOCS_DIR, "fluent-api", "Assembly.md"))


@mcp.resource("cq://docs/fluent-api/Sketch")
def get_sketch_docs() -> str:
    """cq Sketch class documentation - 2D sketching operations."""
    return load_doc_file(os.path.join(DOCS_DIR, "fluent-api", "Sketch.md"))


# ============================================================================
# DIRECT API RESOURCES (SHAPES)
# ============================================================================
# The Direct API provides low-level access to geometric shapes and operations
# through Shape classes and their subclasses.
# ============================================================================


@mcp.resource("cq://docs/direct-api/Shape")
def get_shape_docs() -> str:
    """cq Shape class documentation - base class for all geometric shapes."""
    return load_doc_file(os.path.join(DOCS_DIR, "direct-api", "Shape.md"))


@mcp.resource("cq://docs/direct-api/Vertex")
def get_vertex_docs() -> str:
    """cq Vertex class documentation - represents a point in 3D space."""
    return load_doc_file(os.path.join(DOCS_DIR, "direct-api", "Vertex.md"))


@mcp.resource("cq://docs/direct-api/Edge")
def get_edge_docs() -> str:
    """cq Edge class documentation - represents a 1D curve or line segment."""
    return load_doc_file(os.path.join(DOCS_DIR, "direct-api", "Edge.md"))


@mcp.resource("cq://docs/direct-api/Wire")
def get_wire_docs() -> str:
    """cq Wire class documentation - represents a connected sequence of edges."""
    return load_doc_file(os.path.join(DOCS_DIR, "direct-api", "Wire.md"))


@mcp.resource("cq://docs/direct-api/Face")
def get_face_docs() -> str:
    """cq Face class documentation - represents a 2D surface."""
    return load_doc_file(os.path.join(DOCS_DIR, "direct-api", "Face.md"))


@mcp.resource("cq://docs/direct-api/Shell")
def get_shell_docs() -> str:
    """cq Shell class documentation - represents a collection of connected faces."""
    return load_doc_file(os.path.join(DOCS_DIR, "direct-api", "Shell.md"))


@mcp.resource("cq://docs/direct-api/Solid")
def get_solid_docs() -> str:
    """cq Solid class documentation - represents a 3D solid volume."""
    return load_doc_file(os.path.join(DOCS_DIR, "direct-api", "Solid.md"))


@mcp.resource("cq://docs/direct-api/CompSolid")
def get_compsolid_docs() -> str:
    """cq CompSolid class documentation - represents a composite solid."""
    return load_doc_file(os.path.join(DOCS_DIR, "direct-api", "CompSolid.md"))


@mcp.resource("cq://docs/direct-api/Compound")
def get_compound_docs() -> str:
    """cq Compound class documentation - represents a collection of shapes."""
    return load_doc_file(os.path.join(DOCS_DIR, "direct-api", "Compound.md"))


@mcp.resource("cq://docs/direct-api/Mixin1D")
def get_mixin1d_docs() -> str:
    """cq Mixin1D class documentation - mixin for 1D shape operations."""
    return load_doc_file(os.path.join(DOCS_DIR, "direct-api", "Mixin1D.md"))


@mcp.resource("cq://docs/direct-api/Mixin3D")
def get_mixin3d_docs() -> str:
    """cq Mixin3D class documentation - mixin for 3D shape operations."""
    return load_doc_file(os.path.join(DOCS_DIR, "direct-api", "Mixin3D.md"))


# ============================================================================
# GEOMETRY API RESOURCES
# ============================================================================
# The Geometry API provides geometric primitives like vectors, planes, and
# transformations for working with 3D coordinate systems.
# ============================================================================


@mcp.resource("cq://docs/geometry-api/Vector")
def get_vector_docs() -> str:
    """cq Vector class documentation - 3D vector operations and transformations."""
    return load_doc_file(os.path.join(DOCS_DIR, "geometry-api", "Vector.md"))


@mcp.resource("cq://docs/geometry-api/Matrix")
def get_matrix_docs() -> str:
    """cq Matrix class documentation - 3D transformation matrix operations."""
    return load_doc_file(os.path.join(DOCS_DIR, "geometry-api", "Matrix.md"))


@mcp.resource("cq://docs/geometry-api/Plane")
def get_plane_docs() -> str:
    """cq Plane class documentation - defines a 2D plane in 3D space."""
    return load_doc_file(os.path.join(DOCS_DIR, "geometry-api", "Plane.md"))


@mcp.resource("cq://docs/geometry-api/BoundBox")
def get_boundbox_docs() -> str:
    """cq BoundBox class documentation - axis-aligned bounding box."""
    return load_doc_file(os.path.join(DOCS_DIR, "geometry-api", "BoundBox.md"))


@mcp.resource("cq://docs/geometry-api/Location")
def get_location_docs() -> str:
    """cq Location class documentation - position and orientation in 3D space."""
    return load_doc_file(os.path.join(DOCS_DIR, "geometry-api", "Location.md"))


# ============================================================================
# SELECTORS API RESOURCES
# ============================================================================
# The Selectors API provides classes for selecting specific shapes or features
# from a model based on various criteria (position, type, geometry, etc.).
# ============================================================================


@mcp.resource("cq://docs/selectors-api/Selector")
def get_selector_docs() -> str:
    """cq Selector class documentation - base class for shape selection."""
    return load_doc_file(os.path.join(DOCS_DIR, "selectors-api", "Selector.md"))


@mcp.resource("cq://docs/selectors-api/NearestToPointSelector")
def get_nearesttopointselector_docs() -> str:
    """cq NearestToPointSelector class documentation - selects shapes nearest to a point."""
    return load_doc_file(os.path.join(DOCS_DIR, "selectors-api", "NearestToPointSelector.md"))


@mcp.resource("cq://docs/selectors-api/BoxSelector")
def get_boxselector_docs() -> str:
    """cq BoxSelector class documentation - selects shapes within a bounding box."""
    return load_doc_file(os.path.join(DOCS_DIR, "selectors-api", "BoxSelector.md"))


@mcp.resource("cq://docs/selectors-api/BaseDirSelector")
def get_basedirselector_docs() -> str:
    """cq BaseDirSelector class documentation - base class for direction-based selection."""
    return load_doc_file(os.path.join(DOCS_DIR, "selectors-api", "BaseDirSelector.md"))


@mcp.resource("cq://docs/selectors-api/ParallelDirSelector")
def get_paralleldirselector_docs() -> str:
    """cq ParallelDirSelector class documentation - selects shapes parallel to a direction."""
    return load_doc_file(os.path.join(DOCS_DIR, "selectors-api", "ParallelDirSelector.md"))


@mcp.resource("cq://docs/selectors-api/DirectionSelector")
def get_directionselector_docs() -> str:
    """cq DirectionSelector class documentation - selects shapes in a specific direction."""
    return load_doc_file(os.path.join(DOCS_DIR, "selectors-api", "DirectionSelector.md"))


@mcp.resource("cq://docs/selectors-api/PerpendicularDirSelector")
def get_perpendiculardirselector_docs() -> str:
    """cq PerpendicularDirSelector class documentation - selects shapes perpendicular to a direction."""
    return load_doc_file(os.path.join(DOCS_DIR, "selectors-api", "PerpendicularDirSelector.md"))


@mcp.resource("cq://docs/selectors-api/TypeSelector")
def get_typeselector_docs() -> str:
    """cq TypeSelector class documentation - selects shapes by type (Face, Edge, etc.)."""
    return load_doc_file(os.path.join(DOCS_DIR, "selectors-api", "TypeSelector.md"))


@mcp.resource("cq://docs/selectors-api/RadiusNthSelector")
def get_radiusnthselector_docs() -> str:
    """cq RadiusNthSelector class documentation - selects Nth shape by radius."""
    return load_doc_file(os.path.join(DOCS_DIR, "selectors-api", "RadiusNthSelector.md"))


@mcp.resource("cq://docs/selectors-api/CenterNthSelector")
def get_centernthselector_docs() -> str:
    """cq CenterNthSelector class documentation - selects Nth shape by center position."""
    return load_doc_file(os.path.join(DOCS_DIR, "selectors-api", "CenterNthSelector.md"))


@mcp.resource("cq://docs/selectors-api/DirectionMinMaxSelector")
def get_directionminmaxselector_docs() -> str:
    """cq DirectionMinMaxSelector class documentation - selects min/max shapes in a direction."""
    return load_doc_file(os.path.join(DOCS_DIR, "selectors-api", "DirectionMinMaxSelector.md"))


@mcp.resource("cq://docs/selectors-api/DirectionNthSelector")
def get_directionnthselector_docs() -> str:
    """cq DirectionNthSelector class documentation - selects Nth shape along a direction."""
    return load_doc_file(os.path.join(DOCS_DIR, "selectors-api", "DirectionNthSelector.md"))


@mcp.resource("cq://docs/selectors-api/LengthNthSelector")
def get_lengthnthselector_docs() -> str:
    """cq LengthNthSelector class documentation - selects Nth shape by length."""
    return load_doc_file(os.path.join(DOCS_DIR, "selectors-api", "LengthNthSelector.md"))


@mcp.resource("cq://docs/selectors-api/AreaNthSelector")
def get_areanthselector_docs() -> str:
    """cq AreaNthSelector class documentation - selects Nth shape by area."""
    return load_doc_file(os.path.join(DOCS_DIR, "selectors-api", "AreaNthSelector.md"))


@mcp.resource("cq://docs/selectors-api/BinarySelector")
def get_binaryselector_docs() -> str:
    """cq BinarySelector class documentation - base class for combining selectors."""
    return load_doc_file(os.path.join(DOCS_DIR, "selectors-api", "BinarySelector.md"))


@mcp.resource("cq://docs/selectors-api/AndSelector")
def get_andselector_docs() -> str:
    """cq AndSelector class documentation - logical AND of two selectors."""
    return load_doc_file(os.path.join(DOCS_DIR, "selectors-api", "AndSelector.md"))


@mcp.resource("cq://docs/selectors-api/SumSelector")
def get_sumselector_docs() -> str:
    """cq SumSelector class documentation - combines results from multiple selectors."""
    return load_doc_file(os.path.join(DOCS_DIR, "selectors-api", "SumSelector.md"))


@mcp.resource("cq://docs/selectors-api/SubtractSelector")
def get_subtractselector_docs() -> str:
    """cq SubtractSelector class documentation - subtracts one selector result from another."""
    return load_doc_file(os.path.join(DOCS_DIR, "selectors-api", "SubtractSelector.md"))


@mcp.resource("cq://docs/selectors-api/InverseSelector")
def get_inverseselector_docs() -> str:
    """cq InverseSelector class documentation - inverts a selector result."""
    return load_doc_file(os.path.join(DOCS_DIR, "selectors-api", "InverseSelector.md"))


@mcp.resource("cq://docs/selectors-api/StringSyntaxSelector")
def get_stringsyntaxselector_docs() -> str:
    """cq StringSyntaxSelector class documentation - parses and applies string-based selector syntax."""
    return load_doc_file(os.path.join(DOCS_DIR, "selectors-api", "StringSyntaxSelector.md"))


# ============================================================================
# FREE FUNCTION API RESOURCES
# ============================================================================
# The Free Function API provides standalone functions for creating and
# manipulating shapes without requiring a Workplane context.
# ============================================================================


@mcp.resource("cq://docs/free-functions/all")
def get_free_functions_docs() -> str:
    """
    cq Free Functions documentation - standalone functions from shapes.py.

    These functions provide direct access to shape creation and manipulation
    without needing a Workplane context. Functions include:
    - Shape creation: box, cylinder, sphere, torus, cone
    - 2D primitives: circle, ellipse, rect, polygon, spline
    - Boolean operations: fuse, cut, intersect
    - Transformations: extrude, revolve, sweep, loft
    - Utilities: fillet, chamfer, offset, project, check
    """
    return load_doc_file(os.path.join(DOCS_DIR, "free-functions", "all.md"))


# ============================================================================
# OVERVIEW AND INDEX RESOURCES
# ============================================================================


@mcp.resource("cq://docs/overview")
def get_overview_docs() -> str:
    """cq MCP Overview - comprehensive guide to all available API tools."""
    return load_doc_file(os.path.join(DOCS_DIR, "overview.md"))

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Run the server with stdio transport
    mcp.run(transport="stdio")
