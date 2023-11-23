using Godot;
using System;
using System.Collections.Generic;

[GlobalClass]
public partial class RoadManager : Node
{
	[ExportGroup("Road Materials")]
	[Export] private Material _roadMaterial;
	[Export] private Material _dirtMaterial;
	[ExportGroup("Direction Changes")]
	[Export(PropertyHint.Range, "0.,5.")] private float _directionChangeRangeX = 1.0f;
	[Export(PropertyHint.Range, "0.,5.")] private float _directionChangeRangeY = 1.0f;
	private Vector3 _directionTarget;
	private Vector3 _currentDirection;
	[Export] private Vector2 _directionExtremes = new Vector2(10.0f, 3.0f);
	[Export(PropertyHint.Range, "0.001, 0.1")] private float _directionChangeSpeed = 0.01f;
	[ExportGroup("Road Generation")]
	[Export(PropertyHint.Range, "10f,100f")] private float roadWidth = 50.0f;
	[Export(PropertyHint.Range, "10f,100f")] private float roadLength = 50.0f;
	[Export] private int _roadXSegments = 10;
	[Export] private int _sideXSegments = 10;
	[Export] private int _forwardSegments = 10;
	[Export(PropertyHint.Range, "2,20")] private int totalChunkCount = 10;
	[ExportGroup("Road Movement")]
	[Export(PropertyHint.Range, "0.1,10")] private float _roadSpeed = 1.0f;
	LinkedList<RoadSegment> _roadSegmentsList = new LinkedList<RoadSegment>();

	public override void _Ready()
	{
		for (int i = 0; i < totalChunkCount; i++)
		{
			GenerateSegment();
		}
	}

	public override void _Process(double delta)
	{
		if (_roadSegmentsList.Count == 0)
		{
			return;
		}
		foreach (RoadSegment roadSegment in _roadSegmentsList)
		{
			roadSegment.Move(Vector3.Back * (float)delta * _roadSpeed);
		}
		//check if the first segment is out of view
		if (_roadSegmentsList.First.Value.RoadBody.GlobalTransform.Origin.Z > roadLength)
		{
			//remove the first segment
			_roadSegmentsList.First.Value.RoadBody.QueueFree();
			_roadSegmentsList.RemoveFirst();
			GenerateSegment();
		}
	}

	private void GenerateSegment()
	{
		DirectionStep();
		Vector3 currentOrigin = Vector3.Zero;
		Vector3 currentNormal;
		Vector3 nextNormal;
		Vector3 nextDirection = CalcNextCurrentDirection();
		Vector3 directionOffset;
		float _currentX;
		float sideXSegmentSize = roadWidth * 0.4f / _sideXSegments;
		float roadXSegmentSize = roadWidth * 0.2f / _roadXSegments;
		float ZSegmentSize = roadLength / _forwardSegments;
		float uvX;
		float uvXStepSide = 1f / _sideXSegments;
		float uvXStepRoad = 1f / _roadXSegments;
		float uvY = 0f;
		float uvYStep = 1f / _forwardSegments;
		// create left side
		_currentX = -roadWidth / 2f;
		SurfaceTool st1 = new SurfaceTool();
		st1.Begin(Mesh.PrimitiveType.Triangles);
		st1.SetMaterial(_dirtMaterial);
		SurfaceTool st2 = new SurfaceTool();
		st2.Begin(Mesh.PrimitiveType.Triangles);
		st2.SetMaterial(_roadMaterial);
		SurfaceTool st3 = new SurfaceTool();
		st3.Begin(Mesh.PrimitiveType.Triangles);
		st3.SetMaterial(_dirtMaterial);
		// advance direction one step
		_currentDirection = nextDirection;
		nextDirection = CalcNextCurrentDirection();
		currentNormal = _currentDirection.Cross(Vector3.Right);
		nextNormal = nextDirection.Cross(Vector3.Right);
		directionOffset = _currentDirection.Normalized() * ZSegmentSize;
		// step through forward direction
		for (int yStep = 0; yStep < _forwardSegments; yStep++)
		{
			// left side
			uvX = 0f;
			for (int xStep = 0; xStep < _sideXSegments; xStep++)
			{
				// bottom left
				st1.SetUV(new Vector2(uvX, uvY));
				st1.SetNormal(currentNormal);
				st1.AddVertex(currentOrigin + Vector3.Right * _currentX);
				// top left
				st1.SetUV(new Vector2(uvX, uvY + uvYStep));
				st1.SetNormal(nextNormal);
				st1.AddVertex(currentOrigin + Vector3.Right * _currentX + directionOffset);
				// bottom right
				st1.SetUV(new Vector2(uvX + uvXStepSide, uvY));
				st1.SetNormal(currentNormal);
				st1.AddVertex(currentOrigin + Vector3.Right * (_currentX + sideXSegmentSize));
				// second triangle
				// top right
				st1.SetUV(new Vector2(uvX + uvXStepSide, uvY + uvYStep));
				st1.SetNormal(nextNormal);
				st1.AddVertex(currentOrigin + Vector3.Right * (_currentX + sideXSegmentSize) + directionOffset);
				// bottom right
				st1.SetUV(new Vector2(uvX + uvXStepSide, uvY));
				st1.SetNormal(currentNormal);
				st1.AddVertex(currentOrigin + Vector3.Right * (_currentX + sideXSegmentSize));
				// top left
				st1.SetUV(new Vector2(uvX, uvY + uvYStep));
				st1.SetNormal(nextNormal);
				st1.AddVertex(currentOrigin + Vector3.Right * _currentX + directionOffset);

				_currentX += sideXSegmentSize;
				uvX += uvXStepSide;
			}
			// road
			uvX = 0f;
			for (int x = 0; x < _roadXSegments; x++)
			{
				// bottom left
				st2.SetUV(new Vector2(uvX, uvY));
				st2.SetNormal(currentNormal);
				st2.AddVertex(currentOrigin + Vector3.Right * _currentX);
				// top left
				st2.SetUV(new Vector2(uvX, uvY + uvYStep));
				st2.SetNormal(nextNormal);
				st2.AddVertex(currentOrigin + Vector3.Right * _currentX + directionOffset);
				// bottom right
				st2.SetUV(new Vector2(uvX + uvXStepRoad, uvY));
				st2.SetNormal(currentNormal);
				st2.AddVertex(currentOrigin + Vector3.Right * (_currentX + roadXSegmentSize));

				// second triangle
				// top right
				st2.SetUV(new Vector2(uvX + uvXStepRoad, uvY + uvYStep));
				st2.SetNormal(nextNormal);
				st2.AddVertex(currentOrigin + Vector3.Right * (_currentX + roadXSegmentSize) + directionOffset);
				// bottom right
				st2.SetUV(new Vector2(uvX + uvXStepRoad, uvY));
				st2.SetNormal(currentNormal);
				st2.AddVertex(currentOrigin + Vector3.Right * (_currentX + roadXSegmentSize));
				// top left
				st2.SetUV(new Vector2(uvX, uvY + uvYStep));
				st2.SetNormal(nextNormal);
				st2.AddVertex(currentOrigin + Vector3.Right * _currentX + directionOffset);

				_currentX += roadXSegmentSize;
				uvX += uvXStepRoad;
			}
			// right side
			uvX = 0f;
			for (int x = 0; x < _sideXSegments; x++)
			{
				// bottom left
				st3.SetUV(new Vector2(uvX, uvY));
				st3.SetNormal(currentNormal);
				st3.AddVertex(currentOrigin + Vector3.Right * _currentX);
				// top left
				st3.SetUV(new Vector2(uvX, uvY + uvYStep));
				st3.SetNormal(nextNormal);
				st3.AddVertex(currentOrigin + Vector3.Right * _currentX + directionOffset);
				// bottom right
				st3.SetUV(new Vector2(uvX + uvXStepSide, uvY));
				st3.SetNormal(currentNormal);
				st3.AddVertex(currentOrigin + Vector3.Right * (_currentX + sideXSegmentSize));

				// second triangle
				// top right
				st3.SetUV(new Vector2(uvX + uvXStepSide, uvY + uvYStep));
				st3.SetNormal(nextNormal);
				st3.AddVertex(currentOrigin + Vector3.Right * (_currentX + sideXSegmentSize) + directionOffset);
				// bottom right
				st3.SetUV(new Vector2(uvX + uvXStepSide, uvY));
				st3.SetNormal(currentNormal);
				st3.AddVertex(currentOrigin + Vector3.Right * (_currentX + sideXSegmentSize));
				// top left
				st3.SetUV(new Vector2(uvX, uvY + uvYStep));
				st3.SetNormal(nextNormal);
				st3.AddVertex(currentOrigin + Vector3.Right * _currentX + directionOffset);

				_currentX += sideXSegmentSize;
				uvX += uvXStepSide;
			}
			currentOrigin += _currentDirection.Normalized() * ZSegmentSize;
			_currentDirection = nextDirection;
			nextDirection = CalcNextCurrentDirection();
			directionOffset = _currentDirection.Normalized() * ZSegmentSize;
			currentNormal = _currentDirection.Cross(Vector3.Right);
			nextNormal = nextDirection.Cross(Vector3.Right);
			uvY += uvYStep;
			_currentX = -roadWidth / 2f;
		}
		Mesh leftSideMesh = st1.Commit();
		Mesh roadMesh = st2.Commit();
		Mesh rightSideMesh = st3.Commit();
		RoadSegment rs = new RoadSegment();
		rs.RoadBody = new StaticBody3D
		{
			Name = "RoadSegment|" + rs.Id
		};
		rs.LeftSideCollisionShape = new CollisionShape3D
		{
			Name = "LeftSideCollisionShape",
			Shape = leftSideMesh.CreateTrimeshShape()
		};
		rs.RoadBody.AddChild(rs.LeftSideCollisionShape);
		rs.RoadCollisionShape = new CollisionShape3D
		{
			Name = "RoadCollisionShape",
			Shape = roadMesh.CreateTrimeshShape()
		};
		rs.RoadBody.AddChild(rs.RoadCollisionShape);
		rs.RightSideCollisionShape = new CollisionShape3D
		{
			Name = "RightSideCollisionShape",
			Shape = rightSideMesh.CreateTrimeshShape()
		};
		rs.RoadBody.AddChild(rs.RightSideCollisionShape);
		rs.LeftSideInstance = new MeshInstance3D
		{
			Name = "LeftSideInstance",
			Mesh = leftSideMesh
		};
		rs.RoadBody.AddChild(rs.LeftSideInstance);
		rs.RoadInstance = new MeshInstance3D
		{
			Name = "RoadInstance",
			Mesh = roadMesh
		};
		rs.RoadBody.AddChild(rs.RoadInstance);
		rs.RightSideInstance = new MeshInstance3D
		{
			Name = "RightSideInstance",
			Mesh = rightSideMesh
		};
		rs.RoadBody.AddChild(rs.RightSideInstance);
		rs.EndPoint = currentOrigin;

		// add to scene
		AddChild(rs.RoadBody);
		if (_roadSegmentsList.Count > 0)
		{
			RoadSegment lastSegment = _roadSegmentsList.Last.Value;
			rs.RoadBody.Position = lastSegment.EndPoint + lastSegment.RoadBody.Position;
		}
		else
			rs.RoadBody.Position = Vector3.Zero;
		_roadSegmentsList.AddLast(rs);
	}

	private Vector3 CalcNextCurrentDirection()
	{
		return _currentDirection.MoveToward(_directionTarget, _directionChangeSpeed);
	}

	private void DirectionStep()
	{
		float changeX = (float)GD.RandRange(-_directionChangeRangeX / 2f, _directionChangeRangeX / 2f);
		float newX = _currentDirection.X + changeX;
		newX = Mathf.Clamp(newX, -_directionExtremes.X / 2f, _directionExtremes.X / 2f);
		float changeY = (float)GD.RandRange(-_directionChangeRangeY / 2f, _directionChangeRangeY / 2f);
		float newY = _currentDirection.Y + changeY;
		newY = Mathf.Clamp(newY, -_directionExtremes.Y / 2f, _directionExtremes.Y / 2f);
		_directionTarget = new Vector3(newX, newY, -1f);
	}
}
