"use client";

import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

interface Props {
  ccdCode: string;
  smiles: string;
  onCcdChange: (value: string) => void;
  onSmilesChange: (value: string) => void;
}

export function LigandInput({ ccdCode, smiles, onCcdChange, onSmilesChange }: Props) {
  return (
    <Tabs defaultValue={ccdCode ? "ccd" : "smiles"} className="w-full">
      <TabsList className="h-8">
        <TabsTrigger value="ccd" className="text-xs font-medium">
          CCD Code
        </TabsTrigger>
        <TabsTrigger value="smiles" className="text-xs font-medium">
          SMILES
        </TabsTrigger>
      </TabsList>
      <TabsContent value="ccd" className="mt-2.5">
        <div className="space-y-1.5">
          <Label className="text-xs font-medium text-muted-foreground">
            Chemical Component Dictionary code
          </Label>
          <Input
            value={ccdCode}
            onChange={(e) => onCcdChange(e.target.value.toUpperCase())}
            placeholder="e.g., HEM, ATP, NAD"
            className="font-mono text-xs"
          />
        </div>
      </TabsContent>
      <TabsContent value="smiles" className="mt-2.5">
        <div className="space-y-1.5">
          <Label className="text-xs font-medium text-muted-foreground">SMILES string</Label>
          <Input
            value={smiles}
            onChange={(e) => onSmilesChange(e.target.value)}
            placeholder="e.g., CC(=O)OC1=CC=CC=C1C(=O)O"
            className="font-mono text-xs"
          />
        </div>
      </TabsContent>
    </Tabs>
  );
}
