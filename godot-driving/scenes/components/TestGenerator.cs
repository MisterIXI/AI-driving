using Godot;
using System;

[GlobalClass]
public partial class TestGenerator : MeshInstance3D
{
	[Export] private CollisionShape3D _shape3d;
	[Export] private Material _roadMaterial;
	// Called when the node enters the scene tree for the first time.
	public override void _Ready()
	{
		Mesh _mesh = GenerateMesh();
		_mesh.CreateTrimeshShape();
		_shape3d.Shape = _mesh.CreateTrimeshShape();
		_mesh.SurfaceSetMaterial(0, _roadMaterial);
		// Mesh = GenerateMesh();
		SetDeferred("mesh", _mesh);
		GD.Print("RoadGenerator._Ready");
	}

	// Called every frame. 'delta' is the elapsed time since the previous frame.
	public override void _Process(double delta)
	{
	}

	private Mesh GenerateMesh()
	{
		// clockwise winding order
		SurfaceTool st = new SurfaceTool();

		st.Begin(Mesh.PrimitiveType.Triangles);

		st.SetNormal(new Vector3(0, 1, 0));
		st.SetUV(new Vector2(0, 0));
		// Add vertices for a triangle
		st.AddVertex(new Vector3(0, 0, 0f));
		st.SetUV(new Vector2(0, 1));
		st.AddVertex(new Vector3(0, -3, -10));
		st.SetUV(new Vector2(1, 0));
		st.AddVertex(new Vector3(10, 0, 0));

		// other side of triangle
		st.SetUV(new Vector2(0, 1));
		st.AddVertex(new Vector3(0, -3, -10));
		st.SetUV(new Vector2(1, 1));
		st.AddVertex(new Vector3(10, -3, -10));
		st.SetUV(new Vector2(1, 0));
		st.AddVertex(new Vector3(10, 0, 0));

		// Commit the surface to create the mesh
		Mesh mesh = st.Commit();
		return mesh;
	}
}
