using System;
using Godot;

public class RoadSegment
{
    private static long _idCounter = 0;
    public long Id { get; private set; } = _idCounter++;
    public StaticBody3D RoadBody;
    public MeshInstance3D RoadInstance;
    public MeshInstance3D LeftSideInstance;
    public MeshInstance3D RightSideInstance;
    public CollisionShape3D RoadCollisionShape;
    public CollisionShape3D LeftSideCollisionShape;
    public CollisionShape3D RightSideCollisionShape;
    public Vector3 EndPoint;
    public void Move(Vector3 direction)
    {
        RoadBody.Position += direction;
    }
}